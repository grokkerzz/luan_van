from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, DoubleType, StructType, StructField, StringType, LongType
from module.preprocessor import FilterDuplicateMappingTransformer
import glob
from pyspark.sql.functions import col
from sklearn.model_selection import train_test_split
import numpy as np
from pyspark.sql.functions import round, rand, col, sum as spark_sum, collect_list, udf, array, lit, exp, broadcast, count


def analyze(spark):
    schema = StructType([
        StructField("F_MAMH", StringType()),
        StructField("F_MAMH_new", StringType())
    ])
    mapping_file = "data/preprocess_test/mapping/mapping.csv"
    mapping_df = spark.read \
        .option("header", "true") \
        .option("treatEmptyValuesAsNulls", "true") \
        .option("inferSchema", "true") \
        .option("charset", "UTF-8") \
        .csv(mapping_file, schema=schema)
    mapping_df.show()
    filter_duplicate_transformer = FilterDuplicateMappingTransformer()\
        .setItemCol("F_MAMH")\
        .setValueCol("F_MAMH_new")

    new_mapping_df = filter_duplicate_transformer.transform(mapping_df)

    if new_mapping_df is None:
        print("Conflict mapping")
    else:
        new_mapping_df.coalesce(1)\
            .write\
            .option("header", "true")\
            .csv("mapping")
