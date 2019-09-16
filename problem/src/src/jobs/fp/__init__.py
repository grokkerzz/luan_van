from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, DoubleType, StructType, StructField, StringType, ArrayType
from pyspark.sql.functions import udf, first, expr, concat, col, count, lit, avg, mean as _mean, struct, collect_set
from module.fp_growth import *
import numpy as np

def load_csv(spark, path):
    print("Loading data ...")
    df = spark.read \
        .option("header", "true") \
        .option("treatEmptyValuesAsNulls", "true") \
        .option("inferSchema", "true") \
        .option("charset", "UTF-8") \
        .csv(path)

    df = df.select("MASV1", "F_MAMH", "F_MAKH", "TKET")
    df = df.withColumn("MASV1", df["MASV1"].cast(DoubleType()))
    df = df.withColumn("MASV1", df["MASV1"].cast(IntegerType()))
    df = df.withColumn("TKET", df["TKET"].cast(DoubleType()))
    return df

def analyze(spark):
    #Column's names
    user_col = "MASV1"
    item_col = "F_MAMH"
    grade_col = "TKET"
    input_file_path = ["data/MT/train.csv","data/MT/validation.csv"]

    
    input_predict_data = [
        (1512400, "CO3059", 10.0),
        (1512400, "CO3031", 9.5),
        (1512400, "CO3055", 9.0),
        (1512400, "CO4027", 9.5),
        (1512400, "CO3029", 8.0),
        (1512400, "CO3021", 10.0),
        (1512400, "IM3001", 9.0),
        (1512400, "MT2001", 7.5),
        (1512400, "SP1007", 8.5),
        (1512400, "MT1005", 8.5),
        (1512400, "PH1003", 7.5),
        (1512400, "CO3043", 0.0),
        (1512400, "CO3025", 1.0),
        (1512400, "CO4313", 2.0)
        ]
    schema = StructType([
        StructField("MASV1", IntegerType(), True),
        StructField("F_MAMH", StringType(), True),
        StructField("TKET", DoubleType(), True)])
    inputDF = spark.createDataFrame(input_predict_data, schema)

    #Training and saving model
    train_df = load_csv(spark, input_file_path)
    model = FPGEstimator(spark, user_col, item_col, grade_col, 0.2, 0.7)
    train_df = train_df.select(user_col, item_col)
    transformer = model.fit(train_df)
    model_location = "fp_mt"
    transformer.save(model_location)


    #Loading model
    model_transformer = FPGTransformer.load(spark, model_location)
    result_df = model_transformer.transform(inputDF)
    result_df.show() 
    
