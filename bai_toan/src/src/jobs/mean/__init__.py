from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.sql.types import IntegerType, DoubleType, StructType, StructField, StringType
from module.mf_nbcf import IBCFEstimator, IBCFWithItemFactor, ALSIBCFMeanModel
from pyspark.sql.functions import udf, col, rank, collect_list
import time
from module.ibcf import *
from module.nbcf import *

def analyze(spark):

    spark = SparkSession.builder.appName("BuildModel").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    data_dir = "data/result_K14.csv"
    df_schema = StructType([
        StructField("faculty", StringType(), False),
        StructField("model", StringType(), False),
        StructField("rmse", StringType(), False),
        StructField("mse", StringType(), False),
        StructField("mae", StringType(), False),
    ])

    df = spark.read \
        .option("header", "true") \
        .option("charset", "UTF-8") \
        .schema(df_schema) \
        .csv(data_dir)


