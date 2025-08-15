from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import os

def main():
    # Initialize Spark session
    spark = SparkSession.builder.appName("UpliftModeling").getOrCreate()

    # Define paths relative to this script's parent directory
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    csv_path = os.path.join(base_dir, "data", "ab_test_data.csv")
    parquet_path = os.path.join(base_dir, "data", "ab_test_data.parquet")

    # Load raw CSV data
    df = spark.read.csv(csv_path, header=True, inferSchema=True)

    # Show schema and sample data
    print("Schema of loaded data:")
    df.printSchema()
    print("Sample data:")
    df.show(5)

    # Basic filtering and counts
    df_treated = df.filter(col("test group") == "treatment")
    df_control = df.filter(col("test group") == "control")

    print(f"Treated count: {df_treated.count()}")
    print(f"Control count: {df_control.count()}")

    # Save data as Parquet for efficient columnar storage & faster Spark I/O
    df.write.mode("overwrite").parquet(parquet_path)
    print(f"Data saved as Parquet at: {parquet_path}")

    # Read Parquet back to demonstrate big-data format usage
    parquet_df = spark.read.parquet(parquet_path)
    print(f"Parquet count: {parquet_df.count()}")

    # Stop Spark session
    spark.stop()

if __name__ == "__main__":
    main()