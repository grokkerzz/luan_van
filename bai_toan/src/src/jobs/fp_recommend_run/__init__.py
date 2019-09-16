from module.fp_growth import FPGEstimator, FPGTransformer
from pyspark.sql.functions import lit, col
from module.util import DataUtil, OutputUtil
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.sql.types import IntegerType, DoubleType, StructType, StructField, StringType


def analyze(spark):
    # input_predict_data = [
    #     (1512400, "CO3059", 10.0),
    #     (1512400, "CO3031", 9.5),
    #     (1512400, "CO3055", 9.0),
    #     (1512400, "CO4027", 9.5),
    #     (1512400, "CO3029", 8.0),
    #     (1512400, "CO3021", 10.0),
    #     (1512400, "IM3001", 9.0),
    #     (1512400, "MT2001", 7.5),
    #     (1512400, "SP1007", 8.5),
    #     (1512400, "MT1005", 8.5),
    #     (1512400, "PH1003", 7.5),
    #     (1512400, "CO3043", 0.0),
    #     (1512400, "CO3025", 1.0),
    #     (1512400, "CO4313", 2.0)
    #     ]
    # schema = StructType([
    #     StructField("MASV1", IntegerType(), True),
    #     StructField("F_MAMH", StringType(), True),
    #     StructField("TKET", DoubleType(), True)])

    # inputDF = spark.createDataFrame(input_predict_data, schema)

    data_util = DataUtil(spark)
    list_faculty = ["MT", "BD", "CK", "DC", "DD", "GT", "HC", "MO", "PD", "QL", "UD", "VL", "VP", "XD"]

    estimator = FPGEstimator(spark, data_util.get_user_col(), data_util.get_item_col(), data_util.get_rating_col(), 0.2, 0.8)

    for faculty in list_faculty:
        data_df = data_util.mapping_course(data_util.load_all_df(faculty))\
            .filter(col(data_util.get_rating_col()) >= 5)
        transformer = estimator.fit(data_df.select(data_util.get_user_col(),data_util.get_item_col()))
        # transformer.transform(inputDF).show()
        transformer.save("model/{}/fp".format(faculty))

