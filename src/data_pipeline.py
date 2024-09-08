from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when

def load_data(file_path: str):
    spark = SparkSession.builder.appName("ChurnPrediction").getOrCreate()
    data = spark.read.csv(file_path, header=True, inferSchema=True)
    return data

def preprocess_data(data):
    # Convert categorical variables to numerical values
    data = data.withColumn("Gender", when(col("Gender") == "Male", 1).otherwise(0))
    data = data.withColumn("Churn", when(col("Churn") == "Yes", 1).otherwise(0))
    
    # Handle missing values
    data = data.na.drop()

    # Feature selection
    data = data.select("Tenure", "MonthlyCharges", "TotalCharges", "Gender", "Churn")
    
    return data

def save_data(data, output_path: str):
    data.write.csv(output_path, header=True)

if __name__ == "__main__":
    data = load_data("data/client_data.csv")
    processed_data = preprocess_data(data)
    save_data(processed_data, "data/processed_data.csv")
