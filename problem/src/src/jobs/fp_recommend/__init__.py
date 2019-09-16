from module.fp_growth import FPGEstimator, FPGTransformer
from pyspark.sql.functions import lit, col
from module.util import DataUtil, OutputUtil
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.sql.types import IntegerType, DoubleType, StructType, StructField, StringType


def analyze(spark, user, items, ratings, faculty):
    items_array = items
    ratings_array = ratings
    spark.conf.set("spark.sql.crossJoin.enabled", "true")

    model_location = "model/{}/fp".format(faculty)

    # load data
    data_util = DataUtil(spark)

    new_df = data_util.mapping_course(data_util.create_df_from_new_data(user, items_array, ratings_array, faculty))
    # new_df.show()
    model_transformer = FPGTransformer.load(spark, model_location)

    recommend_df = model_transformer.transform(new_df)

    # recommend_df.show()

    # predict rating recommended items
    db_df = data_util.mapping_course(data_util.load_all_df(faculty))
    # db_df.printSchema()
    input_df = db_df.union(new_df).withColumn("TKET", col("TKET").cast(DoubleType()))
    # input_df.printSchema()

    # index item
    item_indexer = StringIndexer().setInputCol(data_util.get_item_col()).setOutputCol("F_MAMH_index")
    item_indexer_model = item_indexer.fit(input_df)
    input_index_df = item_indexer_model.transform(input_df) \
        .withColumn("F_MAMH_index", col("F_MAMH_index").cast(IntegerType()))

    recommend_df = item_indexer_model.transform(recommend_df)
    # create model
    als = ALS(rank=2, maxIter=15, regParam=0.01, userCol="MASV1",
              itemCol="F_MAMH_index", ratingCol="TKET", coldStartStrategy="drop", nonnegative=True)

    als_model = als.fit(input_index_df)
    output_df = als_model.transform(recommend_df)
    OutputUtil(spark, "MASV1", "F_MAMH", "prediction").output(output_df, user)
    



