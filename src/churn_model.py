from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

def load_data(file_path: str):
    spark = SparkSession.builder.appName("ChurnPrediction").getOrCreate()
    data = spark.read.csv(file_path, header=True, inferSchema=True)
    return data

def train_model(data):
    # Feature engineering
    assembler = VectorAssembler(inputCols=["Tenure", "MonthlyCharges", "TotalCharges", "Gender"], outputCol="features")
    data = assembler.transform(data)
    
    # Split data into training and test sets
    train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)
    
    # Model definition
    lr = LogisticRegression(featuresCol="features", labelCol="Churn")
    
    # Hyperparameter tuning
    paramGrid = ParamGridBuilder() \
                .addGrid(lr.regParam, [0.01, 0.1, 0.5]) \
                .addGrid(lr.maxIter, [10, 50, 100]) \
                .build()
    
    # Cross-validation
    crossval = CrossValidator(estimator=lr,
                              estimatorParamMaps=paramGrid,
                              evaluator=BinaryClassificationEvaluator(labelCol="Churn"),
                              numFolds=3)
    
    # Train model
    model = crossval.fit(train_data)
    
    # Evaluate model
    predictions = model.transform(test_data)
    evaluator = BinaryClassificationEvaluator(labelCol="Churn")
    accuracy = evaluator.evaluate(predictions)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    
    # Save model
    model.bestModel.write().overwrite().save("models/logistic_regression_model")

if __name__ == "__main__":
    data = load_data("data/processed_data.csv")
    train_model(data)
