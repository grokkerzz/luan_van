from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, DoubleType, StructType, StructField, StringType, ArrayType
from pyspark.sql.functions import lit, col
from module.nbcf import NBCFTransformer
from module.util import DataUtil, OutputUtil


def analyze(spark, user, items, ratings, faculty):
    data_util = DataUtil(spark)
    model_location = "model/{}/nbcf".format(faculty)
    item_df = data_util.load_all_df(faculty).select(data_util.get_item_col()).distinct()

    new_data = data_util.mapping_course(data_util.create_df_from_new_data(user, items, ratings, faculty))

    missing_data = item_df.subtract(new_data.select(data_util.get_item_col()).distinct())\
        .withColumn(data_util.get_user_col(), lit(user).cast(IntegerType()))

    ubcf_model = NBCFTransformer.load(spark, model_location)

    output_df = ubcf_model.transform(missing_data)
    OutputUtil(spark, "MASV1", "F_MAMH", "prediction").output(output_df, user)

