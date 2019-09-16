from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.sql.types import IntegerType, DoubleType, LongType, StructField, StringType, StructType
from pyspark.sql.functions import udf, col, rank, collect_list
import numpy as np
from pyspark.sql import DataFrame
import pyspark.sql.functions as spark_func
from pyspark.sql.window import Window
import glob
from pyspark.ml.pipeline import Transformer, Estimator
from .custom_params import HasUserCol, HasItemCol, HasValueCol, HasRank
import pickle

col_id = "id"
col_feature = "features"
col_id1 = "id1"
col_id2 = "id2"
col_feature1 = "features1"
col_feature2 = "features2"
col_similarity = "similarity"


def cosine_similarity(item1, item2):
    dot_product = np.linalg.norm(item1) * np.linalg.norm(item2)
    if dot_product == 0:
        return 0.0
    return float(np.dot(item1, item2) / dot_product)


def euclide_similarity(item1, item2):
    return np.sum(np.power(np.subtract(item1,item2), 2)).tolist()


def weighted_mean(list_score, list_similarity):
    if np.sum(list_similarity) == 0:
       return 0.0
    return (np.multiply(list_score, list_similarity).sum() / np.sum(list_similarity)).tolist()


# class IBCFEstimator(HasUserCol, HasItemCol, HasValueCol, HasRank, Estimator):
#     def __init__(self, spark):
#         super(IBCFEstimator, self).__init__()
#         self.cosine_similarity_udf = udf(cosine_similarity, DoubleType())
#         self.spark = spark
#
#     def _fit(self, item_factor):
#         # print(self.getUserCol())
#         item_df1 = item_factor.withColumnRenamed(col_id, col_id1).withColumnRenamed(col_feature, col_feature1)
#         item_df2 = item_factor.withColumnRenamed(col_id, col_id2).withColumnRenamed(col_feature, col_feature2)
#         item_df1 = self.spark.createDataFrame(item_df1.rdd, item_df1.schema)
#         item_df2 = self.spark.createDataFrame(item_df2.rdd, item_df2.schema)
#         item_similarity_df = item_df1.join(item_df2, item_df1[col_id1] != item_df2[col_id2])
#         item_similarity_df = item_similarity_df.select(col_id1, col_id2,
#                                                        self.cosine_similarity_udf(col(col_feature1), col(col_feature2)).alias(col_similarity))
#         return IBCFTransformer(item_similarity_df)\
#             .setUserCol(self.getUserCol())\
#             .setItemCol(self.getItemCol())\
#             .setValueCol(self.getValueCol())\
#             .setRank(self.getRank())
#
#
# class IBCFTransformer(HasUserCol, HasItemCol, HasValueCol, HasRank, Transformer):
#
#     def __init__(self, item_similarity_df):
#         super(IBCFTransformer, self).__init__()
#         self.item_similarity_df = item_similarity_df
#         self.item_similarity_df.show()
#         self.item_df = item_similarity_df.select(col(col_id1).alias(col_id)).distinct()
#         self.weighted_mean_udf = udf(weighted_mean, DoubleType())
#
#     def _transform(self, dataset):
#         missing_item_col = "missing_id"
#         self.item_df.printSchema()
#         dataset.printSchema()
#         print(self.item_df.crossJoin(dataset.select(self.getUserCol()).distinct()))
#         missing_item_df = self.item_df.crossJoin(dataset.select(self.getUserCol()).distinct())\
#             .withColumnRenamed(col_id, self.getItemCol()).withColumn(self.getItemCol(), col(self.getItemCol()).cast(IntegerType()))\
#             .subtract(dataset.select(self.getUserCol(), self.getItemCol()).distinct())
#         print(missing_item_df.count())
#         missing_item_df.show()
#         window = Window.partitionBy([col(self.getUserCol()), col(missing_item_col)]).orderBy(col(col_similarity).desc())
#         #similarity_score_df (self.getUserCol(), self.getItemCol(), col_id2, self.getValueCol(), col_similarity)
#         similarity_score_df = dataset.join(self.item_similarity_df.withColumnRenamed(col_id1, self.getItemCol()),
#                                            [self.getItemCol()])
#         similarity_score_df.show()
#         a = missing_item_df.join(similarity_score_df.withColumnRenamed(col_id2, missing_item_col), [self.getUserCol(), missing_item_col])
#         a.show()
#         a = a\
#             .select("*", rank().over(window).alias("rank"))
#         a = a\
#             .filter(col("rank") <= self.getRank())
#         a = a\
#             .groupby(self.getUserCol(), missing_item_col) \
#             .agg(collect_list(self.getValueCol()).alias("list_score"),
#                  collect_list(col_similarity).alias("list_similarity"))
#         a.show()
#         a = a\
#             .withColumn(self.getValueCol(), self.weighted_mean_udf(col("list_score"), col("list_similarity")))\
#             .select(self.getUserCol(), col(missing_item_col).alias(self.getItemCol()), self.getValueCol())
#         a.show()
#         return a


class ALSIBCFMeanModel(HasUserCol, HasItemCol, HasValueCol, HasRank, Transformer):
    def __init__(self, spark, ibcf_als, als):
        super(ALSIBCFMeanModel, self).__init__()
        self.als = als
        self.spark = spark
        self.ibcf_als_model = ibcf_als

    def transform(self, input_df, predict_course_df):
        ibcf_predict = self.ibcf_als_model.transform(input_df, predict_course_df)\
            .withColumnRenamed("prediction", "ibcf_prediction")\
            .select(self.getUserCol(), self.getItemCol(), "ibcf_prediction")
        als_predict = self.als.transform(predict_course_df).withColumnRenamed("prediction", "als_prediction")\
            .select(self.getUserCol(), self.getItemCol(), "als_prediction")

        return ibcf_predict.join(als_predict, [self.ibcf_als_model.getUserCol(), self.ibcf_als_model.getItemCol()]) \
            .withColumn("prediction", (col("ibcf_prediction") + col("als_prediction")) / 2)

    def _transform(self, dataset):
        pass


class IBCFWithItemFactor(HasUserCol, HasItemCol, HasValueCol, HasRank, Transformer):
    local_item_index = "IBCFWithItemFactor_local_item_index"
    local_item = "IBCFWithItemFactor_local_item"

    similarity_file_name = "similarity"
    index_file_name = "index"
    param_file_name = "param.dat"

    user_col_save = "user_col"
    item_col_save = "item_col"
    value_col_save = "value_col"
    rank_save = "ran"

    @classmethod
    def create_item_index(cls, df, item_col, index_col):
        return df.select(item_col, index_col)\
            .withColumnRenamed(index_col, IBCFWithItemFactor.local_item_index)\
            .withColumnRenamed(item_col, IBCFWithItemFactor.local_item)\
            .distinct()

    def save(self, location):
        self.item_similarity.rdd.saveAsPickleFile(location + "/" + IBCFWithItemFactor.similarity_file_name, 3)
        self.item_index_df.rdd.saveAsPickleFile(location + "/" + IBCFWithItemFactor.index_file_name, 3)
        model_params = {
            IBCFWithItemFactor.user_col_save: self.getUserCol(),
            IBCFWithItemFactor.item_col_save: self.getItemCol(),
            IBCFWithItemFactor.value_col_save: self.getValueCol(),
            IBCFWithItemFactor.rank_save: self.getRank()
        }
        with open(location + '/param.dat', 'w+') as model_params_file:
            pickle.dump(model_params, model_params_file)

    def load(self, location):
        pickle_similarity_rdd = self.spark.sparkContext.pickleFile(location + "/" + IBCFWithItemFactor.similarity_file_name).collect()
        pickle_index_rdd = self.spark.sparkContext.pickleFile(location + "/" + IBCFWithItemFactor.index_file_name).collect()
        self.item_similarity = self.spark.createDataFrame(pickle_similarity_rdd)
        self.item_index_df = self.spark.createDataFrame(pickle_index_rdd)
        with open(location + '/param.dat', 'rb') as config_dictionary_file:
            model_params = pickle.load(config_dictionary_file)
        self.setUserCol(model_params[IBCFWithItemFactor.user_col_save])
        self.setItemCol(model_params[IBCFWithItemFactor.item_col_save])
        self.setValueCol(model_params[IBCFWithItemFactor.value_col_save])
        self.setRank(model_params[IBCFWithItemFactor.rank_save])
        return self

    def __init__(self, spark, item_factor=None, item_index_df=None):
        super(IBCFWithItemFactor, self).__init__()
        self.cosine_similarity_udf = udf(cosine_similarity, DoubleType())
        self.weighted_mean_udf = udf(weighted_mean, DoubleType())
        if item_factor is not None:
            item_factor.createOrReplaceTempView("ItemFactor")
            item_similarity = spark.sql(
            "SELECT I1.id as id1, I2.id as id2, I1.features as features1, I2.features as features2  FROM ItemFactor I1, ItemFactor I2 WHERE I1.id != I2.id")
            self.item_similarity = item_similarity.withColumn("similarity",
                                                         self.cosine_similarity_udf(item_similarity["features1"],
                                                                                item_similarity["features2"]))

        self.spark = spark
        self.item_index_df = item_index_df # this df must have 2 column local_item and local_item_index, need to rename if change item_col

    def transform(self, input_df, predict_course_df):
        localspace_item_col = IBCFWithItemFactor.local_item_index
        userspace_item_col = IBCFWithItemFactor.local_item
        #item to local_item_index
        input_df = input_df.withColumnRenamed(self.getItemCol(), userspace_item_col)
        predict_course_df = predict_course_df.withColumnRenamed(self.getItemCol(), userspace_item_col)
        if self.item_index_df is not None:
            input_df = self.item_index_df.join(input_df, [userspace_item_col])
            predict_course_df = predict_course_df.join(self.item_index_df, [userspace_item_col])
        else:
            input_df = input_df.withColumn(localspace_item_col, col(userspace_item_col))
            predict_course_df = predict_course_df.withColumn(localspace_item_col, col(userspace_item_col))

        #get similarity score
        similarity_score_df = input_df.join(self.item_similarity,
                                            input_df[localspace_item_col] == self.item_similarity['id1']) \
            .select(self.getUserCol(), self.getValueCol(), 'id1', 'id2', 'similarity') \

        #map local index

        window = Window.partitionBy(
            [col(self.getUserCol()), col(localspace_item_col)]).orderBy(
            col('similarity').desc())

        predict_course_df_predict = predict_course_df.join(
            similarity_score_df.withColumnRenamed("id2", localspace_item_col),
            [localspace_item_col, self.getUserCol()]) \
            .select("*", spark_func.rank().over(window).alias("rank")) \
            .filter(spark_func.col("rank") <= self.getRank())\
            .groupby(self.getUserCol(), localspace_item_col) \
            .agg(collect_list(self.getValueCol()).alias("list_score"),
                 collect_list("similarity").alias("list_similarity"))\
            .withColumn("prediction", self.weighted_mean_udf(col("list_score"), col("list_similarity")))\

        #mapping back item
        if self.item_index_df is not None:
            predict_course_df_predict = predict_course_df_predict\
                .join(self.item_index_df, [localspace_item_col])\
                .withColumnRenamed(userspace_item_col, self.getItemCol())\
                .drop(localspace_item_col)
        else:
            predict_course_df_predict = predict_course_df_predict \
                .withColumnRenamed(localspace_item_col, self.getItemCol())
        return predict_course_df_predict

    def _transform(self, dataset):
        pass

class Predictor(object):

    def __init__(self, spark, user_col_name, item_col_name, rating_col_name, rank=15, maxIter=15,regParam=0.01):
        self.user_col_name = user_col_name
        self.item_col_name = item_col_name
        self.item_col_name_index = "INDEX_" + item_col_name
        self.rating_col_name = rating_col_name
        self.als = ALS(rank=rank, maxIter=maxIter, regParam=regParam, userCol=user_col_name,
                       itemCol=self.item_col_name_index, ratingCol=rating_col_name, coldStartStrategy="drop", nonnegative=True)
        self.item_indexer = StringIndexer().setInputCol(self.item_col_name).setOutputCol(self.item_col_name_index)
        self.item_index_df = None
        self.indexer_model = None
        self.model = None
        self.item_similarity = None
        self.spark = spark

    # fit all the course index
    def fit_item_index(self, item_df):
        self.indexer_model = self.item_indexer.fit(item_df)
        self.item_index_df = self.indexer_model.transform(item_df.select(self.item_col_name).distinct())

    # fit training data (call this after fit_item_index)
    def fit(self, training_df):
        encoded_df = self.indexer_model.transform(training_df)
        # encoded_df = encoded_df.withColumn(self.user_col_name, encoded_df[self.user_col_name].cast(IntegerType()))
        # encoded_df = encoded_df.withColumn(self.rating_col_name, encoded_df[self.rating_col_name].cast(DoubleType()))
        normalize_rating_udf = udf(lambda p: 0.0 if p > 10 else p, DoubleType())
        encoded_df = encoded_df.withColumn(self.rating_col_name, normalize_rating_udf(encoded_df[self.rating_col_name]))
        self.model = self.als.fit(encoded_df)
        item_factor = self.model.itemFactors
        item_factor.createOrReplaceTempView("ItemFactor")

        # function to calculate cosine similarity between two array
        def cosine_similarity(item1, item2):
            dot_product = np.linalg.norm(item1) * np.linalg.norm(item2)
            if dot_product == 0:
                return 0.0
            return float(np.dot(item1, item2) / dot_product)

        cosine_similarity_udf = udf(cosine_similarity, DoubleType())
        item_similarity = self.spark.sql(
            "SELECT I1.id as id1, I2.id as id2, I1.features as features1, I2.features as features2  FROM ItemFactor I1, ItemFactor I2 WHERE I1.id != I2.id")
        self.item_similarity = item_similarity.withColumn("similarity",
                                                          cosine_similarity_udf(item_similarity["features1"],
                                                                                item_similarity["features2"]))

    # self.item_similarity.show()
    # can drop 2 feature column and tempView
    # item_similarity = item_similarity.drop("features1")
    # item_similarity = item_similarity.drop("features2")
    # self.spark.catalog.dropTempView("ItemFactor")

    # input_df will have 1 student id and all course that the student already studied
    # first we will index all the course the student already studied and normalize all score
    # then map similarity data to the already studied course
    # then check if predict_course_df is None or not, if it None, then predict all the remaining course,
    # if not transform the predict_course_df to get the index of predict course
    # then begin predict function (use first 5 relevant course to that course that the student already studied)
    def predict_using_cosine_similarity(self, input_df, predict_course_df=None):
        # preprocessed input data
        # print("begin predict using cosine similarity")
        encoded_df = self.indexer_model.transform(input_df)
        normalize_rating_udf = udf(lambda p: 0.0 if p > 10 else p, DoubleType())
        encoded_df = encoded_df.withColumn(self.rating_col_name, normalize_rating_udf(encoded_df[self.rating_col_name]))

        # get predict course df (remaining course)
        if predict_course_df is None:
            predict_course_df_predict = encoded_df.join(self.item_index_df,
                                                        encoded_df[self.item_col_name_index] != self.item_index_df[
                                                            self.item_col_name_index]) \
                .select(self.user_col_name, self.item_col_name_index)
        else:
            predict_course_df = self.indexer_model.transform(predict_course_df)
            predict_course_df_predict = predict_course_df.drop(self.rating_col_name)

        # get all value that can participate in evaluate final score
        similarity_score_df = encoded_df.join(self.item_similarity,
                                              encoded_df[self.item_col_name_index] == self.item_similarity['id1']) \
            .select(self.user_col_name, self.rating_col_name, 'id1', 'id2', 'similarity') \
            # .withColumnRenamed(self.user_col_name, "user_name_similarity")

        #                encoded_df[self.item_col_name_index] == self.item_similarity['id2']) # can delete this part if allow duplicate id1,id2

        # def predict(student, course, similarity_score_df):
        #     # get first 5 course the student already attended which are the most relevant to the current course
        #     relevant_df = similarity_score_df.filter(similarity_score_df[self.user_col_name] == student and
        #                                              similarity_score_df['id2'] == course) \
        #         .orderBy('similarity', ascending=False) \
        #         .head(5)
        #     relevant_df = relevant_df.withColumn('score', relevant_df[self.rating_col_name] * relevant_df['similarity'])
        #     return relevant_df.select(spark_func.avg(relevant_df['score']).alias('avg')).collect()[0][
        #         'avg']  # need to check again if avg is enough
        def predict(list_score, list_similarity):
            sum_simi = sum(list_similarity)
            if sum_simi == 0:
                return 0.0
            return sum([list_score[i] * list_similarity[i] for i in range(len(list_score))]) / sum(list_similarity)

        predict_udf = udf(predict, DoubleType())
        window = Window.partitionBy(
            [spark_func.col(self.user_col_name), spark_func.col(self.item_col_name_index)]).orderBy(
            spark_func.col('similarity').desc())

        predict_course_df_predict = predict_course_df_predict.join(
            similarity_score_df.withColumnRenamed("id2", self.item_col_name_index),
            [self.item_col_name_index, self.user_col_name]) \
            .select("*", spark_func.rank().over(window).alias("rank")) \
            .filter(spark_func.col("rank") <= 7).groupby(self.user_col_name, self.item_col_name_index) \
            .agg(spark_func.collect_list(self.rating_col_name).alias("list_score"),
                 spark_func.collect_list("similarity").alias("list_similarity"))
        predict_course_df_predict = predict_course_df_predict.withColumn("prediction",
                                                                         predict_udf(spark_func.col("list_score"),
                                                                                     spark_func.col("list_similarity")))

        if predict_course_df is not None and self.rating_col_name in predict_course_df.columns:
            predict_course_df_predict = predict_course_df_predict.join(predict_course_df,
                                                                       [self.user_col_name, self.item_col_name_index])

        return predict_course_df_predict

    def transform(self, df):
        encoded_df = self.indexer_model.transform(df)
        normalize_rating_udf = udf(lambda p: 0.0 if p > 10 else p, DoubleType())
        encoded_df = encoded_df.withColumn(self.rating_col_name, normalize_rating_udf(encoded_df[self.rating_col_name]))
        return self.model.transform(encoded_df)


def load_csv(spark, path):
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
    # df = df.withColumn("TKET", df["TKET"] * 4 / 10)
    # mean = df.select(_mean(col("TKET")).alias("mean")).collect()[0]["mean"]
    # df = df.withColumn("mean", lit(mean))
    # df = df.withColumn("TKET", col("TKET") - col("mean"))
    return df


def test_colla(spark, train, test_input, test_output, item_df, rank_list):
    evaluator_rmse = RegressionEvaluator(metricName="rmse", labelCol="TKET",
                                         predictionCol="prediction")
    evaluator_mse = RegressionEvaluator(metricName="mse", labelCol="TKET",
                                       predictionCol="prediction")
    evaluator_mae = RegressionEvaluator(metricName="mae", labelCol="TKET",
                                       predictionCol="prediction")
    evaluators = [evaluator_rmse, evaluator_mse, evaluator_mae]
    error_list = {}
    for i in range(len(rank_list)):
        predict_model = Predictor(spark, "MASV1", "F_MAMH", "TKET", rank=rank_list[i])
        predict_model.fit_item_index(item_df)
        predict_model.fit(train)

        predicted = predict_model.predict_using_cosine_similarity(test_input, test_output)

        # predicted  = predicted.withColumn("TKET", col("TKET") + col("mean")).withColumn("prediction", col("prediction") + col("mean"))

        error = [evaluator.evaluate(predicted) for evaluator in evaluators]

        error_list[rank_list[i]] = error

    return error_list


def test_MF(spark, train, test_input, test_output, item_df, rank_list):
    evaluator_rmse = RegressionEvaluator(metricName="rmse", labelCol="TKET",
                                         predictionCol="prediction")
    evaluator_mse = RegressionEvaluator(metricName="mse", labelCol="TKET",
                                       predictionCol="prediction")
    evaluator_mae = RegressionEvaluator(metricName="mae", labelCol="TKET",
                                       predictionCol="prediction")
    evaluators = [evaluator_rmse, evaluator_mse, evaluator_mae]
    error_list = {}
    for i in range(len(rank_list)):
        predict_model = Predictor(spark, "MASV1", "F_MAMH", "TKET", rank=rank_list[i])
        predict_model.fit_item_index(item_df)
        predict_model.fit(train.unionAll(test_input))

        predicted = predict_model.transform(test_output)
        # predicted  = predicted.withColumn("TKET", col("TKET") + col("mean")).withColumn("prediction", col("prediction") + col("mean"))

        error = [evaluator.evaluate(predicted) for evaluator in evaluators]

        error_list[rank_list[i]] = error

    return error_list

if __name__ == "__main__":
    # allSubDir = glob.glob("data/")
    # allcsv = []
    # for subdir in allSubDir:
    #     files = glob.glob(subdir + "*.csv")
    #     allcsv = allcsv + files
    # input_file = allcsv
    train_all_data_path = "train_all_data.csv"
    train_MT_data_path = "train_MT_data.csv"
    test_MT_input_data_path = "test_MT_input_data.csv"
    test_MT_output_data_path = "test_MT_output_data.csv"

    spark = SparkSession.builder.appName("BuildModel").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    # df = spark.read.format("com.crealytics.spark.excel").option("location", input_file) \
    #     .option("useHeader", "True") \
    #     .option("treatEmptyValuesAsNulls", "true") \
    #     .option("inferSchema", "False") \
    #     .option("addColorColumns", "False") \
    #     .load()  # original input file
    train_all_df = load_csv(spark, train_all_data_path)

    train_MT_df = load_csv(spark, train_MT_data_path)

    test_MT_input_data = load_csv(spark, test_MT_input_data_path)

    test_MT_output_data = load_csv(spark, test_MT_output_data_path)
    # all_SV_df = train_all_df.unionAll(test_MT_output_data).unionAll(test_MT_input_data).select("MASV1").distinct()
    # print(all_SV_df.count())
    all_MH_df = train_all_df.unionAll(test_MT_output_data).unionAll(test_MT_input_data).select("F_MAMH").distinct()

    rank_list_als = [3, 5, 7, 9]

    all_model_error = test_colla(spark, train_all_df, test_MT_input_data, test_MT_output_data, all_MH_df, rank_list_als)
    MT_model_error = test_colla(spark, train_MT_df, test_MT_input_data, test_MT_output_data, all_MH_df, rank_list_als)
    all_model_MF_error = test_MF(spark, train_all_df, test_MT_input_data, test_MT_output_data, all_MH_df, rank_list_als)
    MT_model_MF_error = test_MF(spark, train_MT_df, test_MT_input_data, test_MT_output_data, all_MH_df, rank_list_als)

    print("model,rank,rmse,mse,mae")
    for key, value in all_model_error.items():
        error_string = "all_Colla" + "," + str(key) + "," + ",".join(str(error) for error in value)
        print(error_string)

    for key, value in MT_model_error.items():
        error_string = "MT_Colla" + "," + str(key) + "," + ",".join(str(error) for error in value)
        print(error_string)

    for key, value in all_model_MF_error.items():
        error_string = "all_MF" + "," + str(key) + "," + ",".join(str(error) for error in value)
        print(error_string)

    for key, value in MT_model_MF_error.items():
        error_string = "MT_MF" + "," + str(key) + "," + ",".join(str(error) for error in value)
        print(error_string)

    # predicted.show()
    # evaluator_rmse = RegressionEvaluator(metricName="rmse", labelCol="TKET",
    #                                      predictionCol="prediction")
    #
    # evaluator_r2 = RegressionEvaluator(metricName="r2", labelCol="TKET",
    #                                    predictionCol="prediction")

    # all_predicted = all_predict_model.predict_using_cosine_similarity(test_MT_input_data, test_MT_output_data)
    # MT_predicted = MT_predict_model.predict_using_cosine_similarity(test_MT_input_data, test_MT_output_data)



    # predicted.show()
    # all_rmse = evaluator_rmse.evaluate(all_predicted)
    # print("Root-mean-square error = " + str(all_rmse))
    #
    # all_r2 = evaluator_r2.evaluate(all_predicted)
    # print("R2 error = " + str(all_r2))
    #
    # # predicted.show()
    # MT_rmse = evaluator_rmse.evaluate(MT_predicted)
    # print("Root-mean-square error = " + str(MT_rmse))
    #
    # MT_r2 = evaluator_r2.evaluate(MT_predicted)
    # print("R2 error = " + str(MT_r2))

