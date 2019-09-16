from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, DoubleType, StructType, StructField, StringType, ArrayType
from module.nbcf import *
import numpy as np
import matplotlib.pyplot as plt

#load data from file
def load_csv(spark, path):
    print("Loading data ...")
    df = spark.read \
        .option("header", "true") \
        .option("treatEmptyValuesAsNulls", "true") \
        .option("inferSchema", "true") \
        .option("charset", "UTF-8") \
        .csv(path)

    df = df.select("MASV1", "F_MAMH", "F_MAKH", "TKET")
    # df = df.filter(df["F_MAKH"] == "MT")
    # print(df.count())
    df = df.withColumn("MASV1", df["MASV1"].cast(DoubleType()))
    df = df.withColumn("MASV1", df["MASV1"].cast(IntegerType()))
    df = df.withColumn("TKET", df["TKET"].cast(DoubleType()))
    return df

   

def analyze(spark):
    spark = SparkSession.builder.appName("NBCF").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    input_file_path = ["data/MT/train.csv","data/MT/test.csv","data/MT/validation.csv"]

    user_col = "MASV1"
    item_col = "F_MAMH"
    grade_col = "TKET"
    prediction_col = "prediction"

    # *******************LOADING DATA*******************************************
    train_df = load_csv(spark, input_file_path[0])
    validate_df = load_csv(spark, input_file_path[2])

    # *******************FOR TRAINING AND SAVING MODEL*******************************************
    rank = 15
    model = NBCFEstimator(spark, user_col, item_col, grade_col, prediction_col, rank)
   
 
    (train_part_df, validate_part_df) = validate_df.randomSplit([0.5, 0.5])
    train_part_df = spark.createDataFrame(train_part_df.rdd, train_part_df.schema)

    train_df = train_df.unionAll(train_part_df)
    # indexer_df = model.create_indexer_df(train_df)
    # train_df = model.index_item(train_df, indexer_df)

    transformed_df = train_df.select(user_col, item_col, grade_col) 
    # transformed_df.show()
    transformed_df.cache()

    #Save model
    nbcf_estimator = model.fit(transformed_df)
    nbcf_estimator.save_model("nbcf_model")
    
    # **********************FOR LOADING AND TRANSFORMING MODEL***********************  
    #Load model to transform 
    nbcf_model = NBCFTransformer.load_model(spark, "nbcf_model")
    result_df = nbcf_model.transform(validate_part_df)
    # result_df.show()
    # error = nbcf_model.evaluate_error(result_df)
    # print(error)        