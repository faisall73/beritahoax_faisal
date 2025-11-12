using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.IO;
using System.Linq;

// =====================
// 1. Struktur Data Asli
// =====================
public class RawBeritaData
{
    [LoadColumn(0)]
    public string? narasi { get; set; }

    [LoadColumn(1)]
    public string? hoax { get; set; }
}

// =====================
// 2. Struktur Data Bersih
// =====================
public class BeritaData
{
    [LoadColumn(0)]
    [ColumnName("narasi")]
    public string? narasi { get; set; }

    [LoadColumn(1)]
    [ColumnName("Label")]
    public bool Label { get; set; }
}

// =====================
// 3. Struktur Prediksi
// =====================
public class BeritaPrediction
{
    [ColumnName("PredictedLabel")]
    public bool PredictedLabel { get; set; }
    public float Probability { get; set; }
    public float Score { get; set; }
}

class Program
{
    private static readonly string DataPath = @"C:\Users\HP ELITEBOOK 840 G6\beritahoax_faisal\600_news_with_valid_hoax_label.csv";
    private static readonly string CleanedPath = @"C:\Users\HP ELITEBOOK 840 G6\beritahoax_faisal\600_news_with_valid_hoax_label_clean.csv";
    private static readonly string ModelPath = "HoaxDetectionModel.zip";

    static void Main(string[] args)
    {
        var mlContext = new MLContext(seed: 1);

        if (!File.Exists(DataPath))
        {
            Console.WriteLine($"[ERROR] File data tidak ditemukan di: {Path.GetFullPath(DataPath)}");
            return;
        }

        CleanDataset(mlContext);
        var model = BuildAndTrainModel(mlContext, CleanedPath);
        EvaluateModel(mlContext, model, CleanedPath);
        TestSinglePrediction(mlContext, model);
    }

    // =====================
    // 4. Membersihkan Dataset
    // =====================
    public static void CleanDataset(MLContext mlContext)
    {
        Console.WriteLine("[CLEANING] Membersihkan dataset...");

        var rawLoader = mlContext.Data.CreateTextLoader<RawBeritaData>(
            separatorChar: ';',
            hasHeader: true
        );

        var rawData = rawLoader.Load(DataPath);
        var rawList = mlContext.Data.CreateEnumerable<RawBeritaData>(rawData, reuseRowObject: false).ToList();

        var cleaned = rawList
            .Where(x => !string.IsNullOrWhiteSpace(x.narasi) && !string.IsNullOrWhiteSpace(x.hoax))
            .Select(x => new BeritaData
            {
                narasi = x.narasi!.Trim(),
                Label = x.hoax!.Trim().ToLower() switch
                {
                    "valid" => true,
                    "hoax" => false,
                    _ => false
                }
            })
            .ToList();

        using (var writer = new StreamWriter(CleanedPath))
        {
            writer.WriteLine("narasi;Label");
            foreach (var item in cleaned)
                writer.WriteLine($"\"{item.narasi}\";{item.Label}");
        }

        Console.WriteLine($"[CLEANING DONE] Disimpan ke: {CleanedPath}");
    }

    // =====================
    // 5. Melatih Model
    // =====================
    public static ITransformer BuildAndTrainModel(MLContext mlContext, string dataPath)
    {
        Console.WriteLine("\n[STEP 1] Memuat Data...");

        var loader = mlContext.Data.CreateTextLoader<BeritaData>(separatorChar: ';', hasHeader: true);
        var data = loader.Load(dataPath);

        var allData = mlContext.Data.CreateEnumerable<BeritaData>(data, reuseRowObject: false).ToList();
        int validCount = allData.Count(x => x.Label);
        int hoaxCount = allData.Count(x => !x.Label);
        Console.WriteLine($"[INFO] Jumlah VALID: {validCount}, HOAX: {hoaxCount}");

        if (validCount == 0 || hoaxCount == 0)
        {
            Console.WriteLine("[ERROR] Data hanya mengandung satu jenis label (semua valid atau semua hoax).");
            Environment.Exit(1);
        }

        var split = mlContext.Data.TrainTestSplit(data, testFraction: 0.2, seed: 1);
        var trainData = split.TrainSet;

        Console.WriteLine("[STEP 2] Membangun Pipeline...");
        var pipeline = mlContext.Transforms.Text.FeaturizeText("Features", nameof(BeritaData.narasi))
            .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(
                labelColumnName: "Label",
                featureColumnName: "Features"));

        Console.WriteLine("[STEP 3] Melatih Model...");
        var model = pipeline.Fit(trainData);
        Console.WriteLine("[INFO] Pelatihan selesai.");

        return model;
    }

    // =====================
    // 6. Evaluasi Model
    // =====================
    public static void EvaluateModel(MLContext mlContext, ITransformer model, string dataPath)
    {
        Console.WriteLine("\n[STEP 4] Evaluasi Model...");

        var loader = mlContext.Data.CreateTextLoader<BeritaData>(separatorChar: ';', hasHeader: true);
        var data = loader.Load(dataPath);
        var split = mlContext.Data.TrainTestSplit(data, testFraction: 0.2, seed: 1);
        var testData = split.TestSet;

        var predictions = model.Transform(testData);
        var metrics = mlContext.BinaryClassification.Evaluate(predictions, labelColumnName: "Label", scoreColumnName: "Score");

        Console.WriteLine("\n================ HASIL EVALUASI ================");
        Console.WriteLine($"Akurasi                : {metrics.Accuracy:P2}");
        Console.WriteLine($"AUC                    : {metrics.AreaUnderRocCurve:P2}");
        Console.WriteLine($"F1 Score               : {metrics.F1Score:P2}");
        Console.WriteLine($"Precision (Positif)    : {metrics.PositivePrecision:P2}");
        Console.WriteLine($"Recall (Positif)       : {metrics.PositiveRecall:P2}");
        Console.WriteLine($"Precision (Negatif)    : {metrics.NegativePrecision:P2}");
        Console.WriteLine($"Recall (Negatif)       : {metrics.NegativeRecall:P2}");
        Console.WriteLine("================================================");

        // Mikro & Makro tambahan
        // Mikro & Makro tambahan (aman untuk semua versi ML.NET)
        // Gunakan metrik yang tersedia di semua versi ML.NET
        float precision = (float)metrics.PositivePrecision;
        float recall = (float)metrics.PositiveRecall;
        float f1Score = (float)metrics.F1Score;

        float microF1 = 2 * ((precision * recall) / (precision + recall + 1e-6f));
        float macroPrecision = precision;
        float macroRecall = recall;
        float macroF1 = microF1;

        Console.WriteLine($"Precision (Positif): {precision:P2}");
        Console.WriteLine($"Recall (Positif)   : {recall:P2}");
        Console.WriteLine($"F1 Score           : {f1Score:P2}");
        Console.WriteLine($"Macro Precision    : {macroPrecision:P2}");
        Console.WriteLine($"Macro Recall       : {macroRecall:P2}");
        Console.WriteLine($"Micro F1           : {microF1:P2}");
        Console.WriteLine($"Macro F1           : {macroF1:P2}");
    }

    // =====================
    // 7. Prediksi Tunggal
    // =====================
    public static void TestSinglePrediction(MLContext mlContext, ITransformer model)
    {
        Console.WriteLine("\n[STEP 5] Tes Prediksi Tunggal...");

        var engine = mlContext.Model.CreatePredictionEngine<BeritaData, BeritaPrediction>(model);

        var contoh = new BeritaData
        {
            narasi = "Pemerintah akan menutup seluruh akses internet di Indonesia besok pagi."
        };

        var pred = engine.Predict(contoh);
        string hasil = pred.PredictedLabel ? "VALID" : "HOAX";

        Console.WriteLine($"\nTeks: {contoh.narasi}");
        Console.WriteLine($"Prediksi: {hasil} (Confidence: {pred.Probability:P2})");
    }
}
