using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace SeismicActivityPrediction
{
    // Data class for seismic activity features
    public class SeismicData
    {
        [LoadColumn(0)]
        public float Latitude { get; set; }

        [LoadColumn(1)]
        public float Longitude { get; set; }

        [LoadColumn(2)]
        public float Depth { get; set; }

        [LoadColumn(3)]
        public float Magnitude { get; set; }

        [LoadColumn(4), ColumnName("Label")]
        public bool Earthquake { get; set; } // True for earthquake, False for no earthquake
    }

    // Class for predictions
    public class SeismicPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool Earthquake { get; set; } // True for earthquake, False for no earthquake

        [ColumnName("Score")]
        public float Probability { get; set; } // Probability of the predicted outcome
    }

    class Program
    {
        static void Main(string[] args)
        {
            // 1. Load the data
            MLContext mlContext = new MLContext();
            string dataPath = "seismic_data.csv"; // Replace with your data file path
            IDataView dataView = mlContext.Data.LoadFromTextFile<SeismicData>(dataPath, hasHeader: true, separatorChar: ',');

            // 2. Define the training pipeline
            var pipeline = mlContext.Transforms.Concatenate("Features", "Latitude", "Longitude", "Depth", "Magnitude")
                .Append(mlContext.BinaryClassification.Trainers.FastTree(labelColumnName: "Label", featureColumnName: "Features"))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel", "PredictedLabel"))
                .Append(mlContext.Transforms.Conversion.MapValueToKey("Score", "Score"));

            // 3. Train the model
            ITransformer model = pipeline.Fit(dataView);

            // 4. Create a prediction engine
            var predictionEngine = mlContext.Model.CreatePredictionEngine<SeismicData, SeismicPrediction>(model);

            // 5. Make a prediction
            SeismicData newSeismicData = new SeismicData()
            {
                // Example seismic data
                Latitude = 34.0522,
                Longitude = -118.2437,
                Depth = 10,
                Magnitude = 4.5
            };

            SeismicPrediction prediction = predictionEngine.Predict(newSeismicData);

            // 6. Display the prediction
            Console.WriteLine($"Predicted Earthquake: {(prediction.Earthquake ? "Yes" : "No")}");
            Console.WriteLine($"Probability: {prediction.Probability}");

            Console.ReadKey();
        }
    }
}
