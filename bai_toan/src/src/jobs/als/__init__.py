from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import lit, col
from module.util import DataUtil, OutputUtil
from pyspark.ml.feature import StringIndexer
from pyspark.sql.types import IntegerType
import json


def analyze(spark, user, items, ratings, faculty):
    items_array = items
    ratings_array = ratings
    # load data
    data_util = DataUtil(spark)

    db_df = data_util.load_all_df(faculty)
    item_df = data_util.get_item_df(db_df)
    # db_df.printSchema()
    new_df = data_util.create_df_from_new_data(user, items_array, ratings_array, faculty)
    # new_df.printSchema()
    new_item_df = data_util.get_item_df(new_df)
    # print(item_df.count())
    input_df = db_df.union(new_df)
    # preprocess
    input_after_mapping_df = data_util.mapping_course(input_df)
    # input_df.show()
    item_df = data_util.mapping_course(item_df)
    # print(item_df.distinct().count())

    new_item_df = data_util.mapping_course(new_item_df)
    # data_util.course_mapper.mapping_df.show()
    # input_df.printSchema()

    # index item
    item_indexer = StringIndexer().setInputCol(data_util.get_item_col()).setOutputCol("F_MAMH_index")
    item_indexer_model = item_indexer.fit(input_after_mapping_df)
    input_index_df = item_indexer_model.transform(input_after_mapping_df) \
        .withColumn("F_MAMH_index", col("F_MAMH_index").cast(IntegerType()))
    # input_index_df.show()
    # input_index_df.printSchema()

    # get missing item
    missing_item_df = item_indexer_model.transform(item_df.subtract(new_item_df)) \
        .withColumn("F_MAMH_index", col("F_MAMH_index").cast(IntegerType())) \
        .withColumn("MASV1", lit(user).cast(IntegerType()))

    # missing_item_df.show()
    # print(item_df.count())
    # print(new_item_df.count())
    # print(missing_item_df.count())
    spark.conf.set("spark.sql.crossJoin.enabled", "true")

    # create model
    als = ALS(rank=2, maxIter=15, regParam=0.01, userCol="MASV1",
              itemCol="F_MAMH_index", ratingCol="TKET", coldStartStrategy="drop", nonnegative=False)

    als_model = als.fit(input_index_df)
    output_df = als_model.transform(missing_item_df)
    # test_df = als_model.transform(input_index_df)
    # test_df.show()
    # output_df.show()
    OutputUtil(spark, "MASV1", "F_MAMH", "prediction").output(output_df, user)
