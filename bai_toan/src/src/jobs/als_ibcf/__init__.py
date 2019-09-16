from module.mf_nbcf import IBCFWithItemFactor
from pyspark.sql.functions import lit, col
from module.util import DataUtil, OutputUtil
from pyspark.ml.feature import StringIndexer
from pyspark.sql.types import IntegerType
import time

import json


def analyze(spark, user, items, ratings, faculty):
    data_util = DataUtil(spark)
    model_location = "model/{}/als_ibcf".format(faculty)
    item_df = data_util.mapping_course(data_util.load_all_df(faculty)).select(data_util.get_item_col()).distinct()
    # start = time.time()
    new_data = data_util.mapping_course(data_util.create_df_from_new_data(user, items, ratings, faculty))
    missing_data = item_df.subtract(new_data.select(data_util.get_item_col()).distinct())\
        .withColumn(data_util.get_user_col(), lit(user).cast(IntegerType()))
    # missing_data.count()
    # end = time.time()
    # print(str(end - start))
    als_ibcf_model = IBCFWithItemFactor(spark)\
        .load(model_location)\
        .setUserCol(data_util.get_user_col())\
        .setItemCol(data_util.get_item_col())\
        .setValueCol(data_util.get_rating_col())

    output_df = als_ibcf_model.transform(new_data, missing_data)
    # print(item_df.
    # input_index_df.show()
    # input_index_df.printSch
    # test_df = als_model.transform(input_index_df)
    # test_df.show()
    # output_df.show()
    OutputUtil(spark, "MASV1", "F_MAMH", "prediction").output(output_df, user)
