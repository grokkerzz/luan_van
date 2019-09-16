from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.sql.types import IntegerType, DoubleType, StructType, StructField, StringType
from module.mf_nbcf import IBCFWithItemFactor, ALSIBCFMeanModel
from pyspark.sql.functions import udf, col, rank, collect_list
import time
from module.ibcf import IBCFEstimator
from module.nbcf import NBCF
from sklearn.model_selection import train_test_split
import numpy as np
import json
from module.baseline_method import MeanTransformer
def load_csv(spark, path):
    df = spark.read \
        .option("header", "true") \
        .option("treatEmptyValuesAsNulls", "true") \
        .option("inferSchema", "true") \
        .option("charset", "UTF-8") \
        .csv(path)
    # df.show()
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

def evaluate(evaluate_df, list_evaluator):
    return [evaluator.evaluate(evaluate_df) for evaluator in list_evaluator]


class Model_Error_Wrapper(object):

    def __init__(self, name, model, error, paramDict):
        self.model = model
        self.error = error
        self.name = name
        self.paramDict = paramDict

    def compare(self, otherModel):
        if self.error > otherModel.error:
            return otherModel
        else:
            return self

    def getModel(self):
        return self.model

    def getName(self):
        return self.name

    def getParams(self):
        return self.paramDict

def put_best_model(best_models, model_name, new_model):
    if model_name in best_models:
        best_models[model_name] = new_model.compare(best_models[model_name])
    else:
        best_models[model_name] = new_model
    return best_models


def get_best_param(spark, train, test_input, test_output, rank_list, ibcf_ranks):
    evaluator_rmse = RegressionEvaluator(metricName="rmse", labelCol="TKET",
                                         predictionCol="prediction")
    evaluator_mse = RegressionEvaluator(metricName="mse", labelCol="TKET",
                                        predictionCol="prediction")
    evaluator_mae = RegressionEvaluator(metricName="mae", labelCol="TKET",
                                        predictionCol="prediction")
    evaluators = [evaluator_rmse, evaluator_mse, evaluator_mae]
    error_list_als = {}
    error_list_als_nn = {}
    error_list_als_ibcf = {}
    error_list_als_nn_ibcf = {}
    error_list_combine = {}
    error_list_combine_nn = {}
    error_list_ibcf = {}
    error_list_nbcf = {}
    error_models = {}
    best_models = {}
    # test_input.show()
    # test_output.show()
    # baseline_model = MeanTransformer(spark)\
    #     .setUserCol("MASV1")\
    #     .setItemCol("F_MAMH_index")\
    #     .setValueCol("TKET")\
    #     .setOutputCol("prediction")
    # predi = baseline_model.transform(test_input, test_output)
    # predi.show()
    user_col = "MASV1"
    item_col = "F_MAMH"
    item_index_col = "F_MAMH_index"
    grade_col = "TKET"
    prediction_col = "prediction"
    #
    #IBCF prediction model
    ibcf_estimator = IBCFEstimator(spark, user_col, item_col, item_index_col, grade_col, prediction_col)
    train_part_df = ibcf_estimator.remove_unknown_item(train, test_input)
    validate_part_df = ibcf_estimator.remove_unknown_item(train, test_output)
    ibcf_model = ibcf_estimator.fit(train.drop(item_col))

    for rank in ibcf_ranks:
        result_df = ibcf_model.transform(train_part_df, validate_part_df, rank)
        # result_df.show()
        error_ibcf = evaluate(result_df, evaluators)
        error_list_ibcf[rank] = error_ibcf
        # print(error_ibcf)
        best_models = put_best_model(best_models, "ibcf",
                                     Model_Error_Wrapper("ibcf_{}".format(rank), ibcf_model, error_ibcf[0], {"rank": rank}))
    #
    # #NBCF prediction model
    # nbcf_model = NBCF(spark, user_col, item_col, item_index_col, grade_col, prediction_col)
    # train_df = train.unionAll(test_input)
    # nbcf_model = nbcf_model.fit(train_df.drop(item_col))
    # for rank in ibcf_ranks:
    #     result_df = nbcf_model.transform(train_df, test_output, rank)
    #     # result_df.show()
    #     error_nbcf = evaluate(result_df,evaluators)
    #     error_list_nbcf[rank] = error_nbcf
    #     best_models = put_best_model(best_models, "nbcf",
    #                                  Model_Error_Wrapper("nbcf_{}".format(rank), nbcf_model, error_nbcf[0], {"rank": rank}))

    # for i in range(len(rank_list)):
    #     als_input = train.unionAll(test_input)
    # #
    # #     # als non negative false
    #     als = ALS(rank=rank_list[i], maxIter=15, regParam=0.01, userCol="MASV1",
    #               itemCol="F_MAMH_index", ratingCol="TKET", coldStartStrategy="drop", nonnegative=False)
    #
    #     als_model = als.fit(als_input)
    #     predict_als = als_model.transform(test_output)
    #
    #     # als non negative true
    #     als_nn = ALS(rank=rank_list[i], maxIter=15, regParam=0.01, userCol="MASV1",
    #                  itemCol="F_MAMH_index", ratingCol="TKET", coldStartStrategy="drop", nonnegative=True)
    #     als_nn_model = als_nn.fit(als_input)
    #     predict_als_nn = als_nn_model.transform(test_output)
    #
        # error_als = evaluate(predict_als, evaluators)
    #     error_als_nn = evaluate(predict_als_nn, evaluators)
    #
    #     error_list_als[rank_list[i]] = error_als
    #     error_list_als_nn[rank_list[i]] = error_als_nn
    #
        # best_models = put_best_model(best_models, "als",
        #                              Model_Error_Wrapper("als_{}".format(rank_list[i]), als_model, error_als[0], {"rank": rank_list[i]}))
    #     best_models = put_best_model(best_models, "als_nn",
    #                                  Model_Error_Wrapper("als_nn_{}".format(rank_list[i]), als_nn_model,
    #                                                      error_als_nn[0], {"rank": rank_list[i]}))
    #
    #     # combine mf_ibcf_model
    #
        # for ibcf_rank in ibcf_ranks:
        #     # als_ibcf
        #     als_ibcf_model = IBCFWithItemFactor(spark, als_model.itemFactors) \
        #         .setUserCol("MASV1") \
        #         .setItemCol("F_MAMH_index") \
        #         .setValueCol("TKET") \
        #         .setRank(ibcf_rank)
        #     predict_als_ibcf = als_ibcf_model.transform(test_input, test_output.drop("TKET"))
        #     predict_als_ibcf_with_gt = predict_als_ibcf.join(test_output, ["MASV1", "F_MAMH_index"])
        #
        #     error_als_ibcf = evaluate(predict_als_ibcf_with_gt, evaluators)
        #     error_list_als_ibcf["{}_{}".format(rank_list[i], ibcf_rank)] = error_als_ibcf
        #     best_models = put_best_model(best_models, "als_ibcf",
        #                                  Model_Error_Wrapper("als_ibcf_{}_{}".format(rank_list[i], ibcf_rank),
        #                                                      als_ibcf_model, error_als_ibcf[0], {"als_rank": rank_list[i],
        #                                                                                          "ibcf_rank": ibcf_rank}))
    #
    #         # als_ibcf_mean
    #         als_ibcf_mean_model = ALSIBCFMeanModel(spark, als_ibcf_model, als_model).setUserCol("MASV1") \
    #             .setItemCol("F_MAMH_index") \
    #             .setValueCol("TKET")
    #         combine = als_ibcf_mean_model.transform(test_input, test_output.drop("TKET")).join(test_output, ["MASV1",
    #                                                                                                          "F_MAMH_index"])
    #         # combine.show()
    #
    #         # combine with als
    #         # combine = predict_als_ibcf.withColumnRenamed("prediction", "prediction_ibcf") \
    #         #     .join(predict_als.withColumnRenamed("prediction", "prediction_als"), ["MASV1", "F_MAMH_index"]) \
    #         #     .withColumn("prediction", (col("prediction_ibcf") + col("prediction_als")) / 2)
    #
    #         error_combine = evaluate(combine, evaluators)
    #         error_list_combine["{}_{}".format(rank_list[i], ibcf_rank)] = error_combine
    #         best_models = put_best_model(best_models, "als_ibcf_mean",
    #                                      Model_Error_Wrapper("als_ibcf_mean_{}_{}".format(rank_list[i], ibcf_rank),
    #                                                          als_ibcf_mean_model, error_combine[0],{"als_rank": rank_list[i],
    #                                                                                              "ibcf_rank": ibcf_rank}))
    #
    #         # als_nn_ibcf
    #         als_nn_ibcf_model = IBCFWithItemFactor(spark, als_nn_model.itemFactors) \
    #             .setUserCol("MASV1") \
    #             .setItemCol("F_MAMH_index") \
    #             .setValueCol("TKET") \
    #             .setRank(ibcf_rank)
    #         predict_als_nn_ibcf = als_nn_ibcf_model.transform(test_input, test_output.drop("TKET"))
    #         predict_als_nn_ibcf_with_gt = predict_als_nn_ibcf.join(test_output, ["MASV1", "F_MAMH_index"])
    #
    #         error_als_nn_ibcf = evaluate(predict_als_nn_ibcf_with_gt, evaluators)
    #         error_list_als_nn_ibcf["{}_{}".format(rank_list[i], ibcf_rank)] = error_als_nn_ibcf
    #         best_models = put_best_model(best_models, "als_nn_ibcf",
    #                                      Model_Error_Wrapper("als_nn_ibcf_{}_{}".format(rank_list[i], ibcf_rank),
    #                                                          als_nn_ibcf_model, error_als_nn_ibcf[0],{"als_rank": rank_list[i],
    #                                                                                              "ibcf_rank": ibcf_rank}))
    #
    #         # als_nn_ibcf_mean
    #         als_nn_ibcf_mean_model = ALSIBCFMeanModel(spark, als_nn_ibcf_model, als_nn_model).setUserCol("MASV1") \
    #             .setItemCol("F_MAMH_index") \
    #             .setValueCol("TKET")
    #         combine_nn = als_nn_ibcf_mean_model.transform(test_input, test_output.drop("TKET")).join(test_output,
    #                                                                                                  ["MASV1",
    #                                                                                                   "F_MAMH_index"])
    #         # combine_nn.show()
    #
    #         # combine with als_nn
    #         # combine_nn = predict_als_nn_ibcf.withColumnRenamed("prediction", "prediction_ibcf") \
    #         #     .join(predict_als_nn.withColumnRenamed("prediction", "prediction_als"), ["MASV1", "F_MAMH_index"]) \
    #         #     .withColumn("prediction", (col("prediction_ibcf") + col("prediction_als")) / 2)
    #
    #         error_combine_nn = evaluate(combine_nn, evaluators)
    #         error_list_combine_nn["{}_{}".format(rank_list[i], ibcf_rank)] = error_combine_nn
    #         best_models = put_best_model(best_models, "als_nn_ibcf_mean",
    #                                      Model_Error_Wrapper("als_nn_ibcf_mean_{}_{}".format(rank_list[i], ibcf_rank),
    #                                                          als_nn_ibcf_mean_model, error_combine_nn[0],{"als_rank": rank_list[i],
    #                                                                                              "ibcf_rank": ibcf_rank}))
    #
    # error_models["als"] = error_list_als
    # error_models["als_nn"] = error_list_als_nn
    # error_models["als_ibcf"] = error_list_als_ibcf
    # error_models["als_nn_ibcf"] = error_list_als_nn_ibcf
    # error_models["als_ibcf_mean"] = error_list_combine
    # error_models["als_nn_ibcf_mean"] = error_list_combine_nn




    # best_models["baseline"] = Model_Error_Wrapper("baseline", baseline_model, 0, {})
    error_models["ibcf"] = error_list_ibcf
    # error_models["nbcf"] = error_list_nbcf
    return error_models, best_models


def run_train(spark, train, test_input, param_dict):
    models = {}

    # als_param = param_dict["als"]
    # als_nn_param = param_dict["als_nn"]
    # als_ibcf_param = param_dict["als_ibcf"]
    # als_nn_ibcf_param = param_dict["als_nn_ibcf"]
    # als_ibcf_mean_param = param_dict["als_ibcf_mean"]
    # als_nn_ibcf_mean_param = param_dict["als_nn_ibcf_mean"]

    user_col = "MASV1"
    item_col = "F_MAMH"
    item_index_col = "F_MAMH_index"
    grade_col = "TKET"
    prediction_col = "prediction"
    #IBCF prediction model
    # print("train count: {}".format(train.count()))
    # print("test_input count: {}".format(test_input.count()))

    ibcf_estimator = IBCFEstimator(spark, user_col, item_col, item_index_col, grade_col, prediction_col)
    ibcf_model = ibcf_estimator.fit(train.drop(item_col))
    #
    # nbcf_estimator = NBCF(spark, user_col, item_col, item_index_col, grade_col, prediction_col)
    # nbcf_model = nbcf_estimator.fit(train.unionAll(test_input).drop(item_col))

    # user_col = "MASV1"
    # item_col = "F_MAMH"
    # item_index_col = "F_MAMH_index"
    # grade_col = "TKET"
    # prediction_col = "prediction"
    #
    # #IBCF prediction model
    # ibcf_model = IBCF(spark, user_col, item_col, item_index_col, grade_col, prediction_col)
    # train_part_df = ibcf_model.remove_unknown_item(train, test_input)
    # validate_part_df = ibcf_model.remove_unknown_item(train, test_output)
    # item_similarity_df = ibcf_model.fit(train.drop(item_col))
    #
    # for rank in ibcf_ranks:
    #     result_df = ibcf_model.predict(validate_part_df, item_similarity_df, train_part_df, rank)
    #     result_df.show()
    #     error_ibcf = evaluate(result_df,evaluators)
    #     error_list_ibcf[rank] = error_ibcf
    #
    # #NBCF prediction model
    # nbcf_model = NBCF(spark, user_col, item_col, item_index_col, grade_col, prediction_col)
    # train_df = train.unionAll(test_input)
    # user_similarity = nbcf_model.fit(train_df.drop(item_col))
    # for rank in ibcf_ranks:
    #     result_df = nbcf_model.predict(test_output, user_similarity, train_df, rank)
    #     result_df.show()
    #     error_nbcf = evaluate(result_df,evaluators)
    #     error_list_nbcf[rank] = error_nbcf






    # als_input = train.unionAll(test_input)
    #
    # als non negative false
    # als = ALS(rank=als_param["rank"], maxIter=15, regParam=0.01, userCol="MASV1",
    #               itemCol="F_MAMH_index", ratingCol="TKET", coldStartStrategy="drop", nonnegative=False)
    #
    # als_model = als.fit(als_input)
    #
    # als_nn = ALS(rank=als_nn_param["rank"], maxIter=15, regParam=0.01, userCol="MASV1",
    #                  itemCol="F_MAMH_index", ratingCol="TKET", coldStartStrategy="drop", nonnegative=True)
    # als_nn_model = als_nn.fit(als_input)
    #
    # # combine mf_ibcf_model
    #
    # als_ibcf
    # als_ibcf_als = ALS(rank=als_ibcf_param["als_rank"], maxIter=15, regParam=0.01, userCol="MASV1",
    #                  itemCol="F_MAMH_index", ratingCol="TKET", coldStartStrategy="drop", nonnegative=False)
    # als_ibcf_als_model = als_ibcf_als.fit(als_input)
    # als_ibcf_model = IBCFWithItemFactor(spark, als_ibcf_als_model.itemFactors) \
    #             .setUserCol("MASV1") \
    #             .setItemCol("F_MAMH_index") \
    #             .setValueCol("TKET") \
    #             .setRank(als_ibcf_param["ibcf_rank"])
    #
    # # als_ibcf_mean
    # als_ibcf_mean_als = ALS(rank=als_ibcf_mean_param["als_rank"], maxIter=15, regParam=0.01, userCol="MASV1",
    #                  itemCol="F_MAMH_index", ratingCol="TKET", coldStartStrategy="drop", nonnegative=False)
    # als_ibcf_mean_als_model = als_ibcf_mean_als.fit(als_input)
    # als_ibcf_mean_ibcf_model = IBCFWithItemFactor(spark, als_ibcf_mean_als_model.itemFactors) \
    #             .setUserCol("MASV1") \
    #             .setItemCol("F_MAMH_index") \
    #             .setValueCol("TKET") \
    #             .setRank(als_ibcf_mean_param["ibcf_rank"])
    #
    # als_ibcf_mean_model = ALSIBCFMeanModel(spark, als_ibcf_mean_ibcf_model, als_ibcf_mean_als_model)\
    #             .setUserCol("MASV1") \
    #             .setItemCol("F_MAMH_index") \
    #             .setValueCol("TKET")
    #
    # # als_nn_ibcf
    # als_nn_ibcf_als = ALS(rank=als_nn_ibcf_param["als_rank"], maxIter=15, regParam=0.01, userCol="MASV1",
    #                  itemCol="F_MAMH_index", ratingCol="TKET", coldStartStrategy="drop", nonnegative=True)
    # als_nn_ibcf_als_model = als_nn_ibcf_als.fit(als_input)
    # als_nn_ibcf_model = IBCFWithItemFactor(spark, als_nn_ibcf_als_model.itemFactors) \
    #             .setUserCol("MASV1") \
    #             .setItemCol("F_MAMH_index") \
    #             .setValueCol("TKET") \
    #             .setRank(als_nn_ibcf_param["ibcf_rank"])
    #
    # # als_nn_ibcf_mean
    # als_ibcf_nn_mean_als = ALS(rank=als_nn_ibcf_mean_param["als_rank"], maxIter=15, regParam=0.01, userCol="MASV1",
    #                  itemCol="F_MAMH_index", ratingCol="TKET", coldStartStrategy="drop", nonnegative=False)
    # als_ibcf_nn_mean_als_model = als_ibcf_nn_mean_als.fit(als_input)
    # als_nn_ibcf_mean_ibcf_model = IBCFWithItemFactor(spark, als_ibcf_nn_mean_als_model.itemFactors) \
    #             .setUserCol("MASV1") \
    #             .setItemCol("F_MAMH_index") \
    #             .setValueCol("TKET") \
    #             .setRank(als_nn_ibcf_mean_param["ibcf_rank"])
    #
    # als_nn_ibcf_mean_model = ALSIBCFMeanModel(spark, als_nn_ibcf_mean_ibcf_model, als_ibcf_nn_mean_als_model)\
    #             .setUserCol("MASV1") \
    #             .setItemCol("F_MAMH_index") \
    #             .setValueCol("TKET")
    #
    # baseline_model = MeanTransformer(spark)\
    #     .setUserCol("MASV1")\
    #     .setItemCol("F_MAMH_index")\
    #     .setValueCol("TKET")\
    #     .setOutputCol("prediction")

    models["ibcf"] = ibcf_model
    # models["nbcf"] = nbcf_model
    # models["als"] = als_model
    # models["als_nn"] = als_nn_model
    # models["als_ibcf"] = als_ibcf_model
    # models["als_nn_ibcf"] = als_nn_ibcf_model
    # models["als_ibcf_mean"] = als_ibcf_mean_model
    # models["als_nn_ibcf_mean"] = als_nn_ibcf_mean_model
    # models["baseline"] = baseline_model

    return models


def analyze(spark):
    # spark = SparkSession.builder.appName("BuildModel").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    # spark.conf.set("spark.sql.pivotMaxValues", '100000')
    data_dir = "data/train_val_test_1"
    list_faculty = ["MT", "BD", "CK", "DC", "DD", "GT", "MO", "PD", "QL", "UD", "VL", "VP", "XD","HC"]
    # list_faculty = ["UD", "VL", "VP", "XD"]
    # list_faculty = ["MT"]
    list_faculty_real = ["MO","XD","HC"]

    file_name = ["train.csv", "test.csv", "validation.csv"]
    number_of_run = 10
    user_schema = StructType([StructField("MASV1", IntegerType())])
    evaluator_rmse = RegressionEvaluator(metricName="rmse", labelCol="TKET",
                                             predictionCol="prediction")
    evaluator_mse = RegressionEvaluator(metricName="mse", labelCol="TKET",
                                            predictionCol="prediction")
    evaluator_mae = RegressionEvaluator(metricName="mae", labelCol="TKET",
                                            predictionCol="prediction")
    evaluators = [evaluator_rmse, evaluator_mse, evaluator_mae]
    rank_list_als = [5, 7]
    ibcf_ranks = [3, 5, 7, 9, 11, 13, 15, 17, 20, 23, 25]
    # spark.conf.set("spark.sql.shuffle.partitions", 400)
    # ibcf_ranks = [15, 17, 20]
    for faculty in list_faculty_real:
        all_other_faculty = []

        for included_faculty in list_faculty:
            if included_faculty != faculty:
                for file in file_name:
                    all_other_faculty.append("{}/{}/{}".format(data_dir, included_faculty,file))

        all_other_faculty_df = load_csv(spark, all_other_faculty)

        train_path = "{}/{}/{}".format(data_dir, faculty, "train.csv")
        test_path = "{}/{}/{}".format(data_dir, faculty, "test.csv")
        validation_path = "{}/{}/{}".format(data_dir, faculty, "validation.csv")

        train_current_df = load_csv(spark, train_path)

        train_df = train_current_df.unionAll(all_other_faculty_df)

        validation_df = load_csv(spark, validation_path)

        test_df = load_csv(spark, test_path)

        all_df = train_df.unionAll(validation_df).unionAll(test_df)

        all_current_df = train_current_df.unionAll(validation_df).unionAll(test_df)


        all_item_df = all_df.select("F_MAMH").distinct()
        item_count = all_item_df.count()
        user_count = all_df.select("MASV1").distinct().count()
        rating_count = all_df.count()
        sparsity = 1 - float(rating_count) / float(user_count * item_count)
        print("---------------{}---------------".format(faculty))
        print("Number of item: {}".format(item_count))
        print("Number of user: {}".format(user_count))
        print("Number of rating: {}".format(rating_count))
        print("Sparsity: {}".format(sparsity))


        # index
        item_indexer = StringIndexer().setInputCol("F_MAMH").setOutputCol("F_MAMH_index")
        indexer_model = item_indexer.fit(all_item_df)
        item_index_df = indexer_model.transform(all_item_df)

        train_df = indexer_model.transform(train_df).withColumn("F_MAMH_index", col("F_MAMH_index").cast(IntegerType()))
        test_df = indexer_model.transform(test_df).withColumn("F_MAMH_index", col("F_MAMH_index").cast(IntegerType()))
        validation_df = indexer_model.transform(validation_df).withColumn("F_MAMH_index",
                                                                      col("F_MAMH_index").cast(IntegerType()))

        validation_input_output = validation_df.randomSplit([0.5, 0.5])
        _, best_param = get_best_param(spark, train_df, validation_input_output[0], validation_input_output[1],rank_list_als,ibcf_ranks)
        best_param_dict = {} # key is type of model, value is dict of param

        print("---------------Best Param {}---------------".format(faculty))
        for type_of_model, model_wrapper in best_param.items():
            best_param_dict[type_of_model] = model_wrapper.getParams()
            print("{}: {}".format(type_of_model, json.dumps(model_wrapper.getParams())))

        print("---------------Test {}---------------".format(faculty))
        print("faculty,model,rmse,mse,mae")
        for i in range(number_of_run):
            user_list = all_current_df.select("MASV1").distinct().rdd.flatMap(lambda x: x).collect()
            # 60% train, 20% validation (10% input, 10% output), 20% test (10% input, 10% output)
            train, validation = train_test_split(np.array(user_list), test_size=0.2)
            train, test = train_test_split(np.array(train), test_size=0.25)
            train_df = all_current_df.join(spark.createDataFrame([[x] for x in train.tolist()], schema=user_schema), ["MASV1"])\
                .unionAll(all_other_faculty_df)
            validation_df = all_current_df.join(spark.createDataFrame([[x] for x in validation.tolist()], schema=user_schema), ["MASV1"])
            test_df = all_current_df.join(spark.createDataFrame([[x] for x in test.tolist()], schema=user_schema), ["MASV1"])

            train_df = indexer_model.transform(train_df).withColumn("F_MAMH_index", col("F_MAMH_index").cast(IntegerType()))
            test_df = indexer_model.transform(test_df).withColumn("F_MAMH_index", col("F_MAMH_index").cast(IntegerType()))
            validation_df = indexer_model.transform(validation_df).withColumn("F_MAMH_index",
                                                                              col("F_MAMH_index").cast(IntegerType()))
            train_df = spark.createDataFrame(train_df.rdd, train_df.schema).cache()
            test_df = spark.createDataFrame(test_df.rdd, test_df.schema).cache()
            validation_df = spark.createDataFrame(validation_df.rdd, validation_df.schema).cache()

            validation_input_output = validation_df.randomSplit([0.5, 0.5])
            test_input_output = test_df.randomSplit([0.5, 0.5])

            models = run_train(spark, train_df.unionAll(validation_df).cache(), test_input_output[0], best_param_dict)

            # for model_name, model_errors in error_map.items():
            #     for key, value in model_errors.items():
            #         error_string = model_name + "," + str(key) + "," + ",".join(str(error) for error in value)
            #         print(error_string)

            for type_of_model, model in models.items():
                if type_of_model == "als" or type_of_model == "als_nn":
                    result_df = model.transform(test_input_output[1])
                elif type_of_model == "ibcf":
                    train_part_df = model.remove_unknown_item(train_df.unionAll(validation_df), test_input_output[0]).cache()
                    validate_part_df = model.remove_unknown_item(train_df.unionAll(validation_df), test_input_output[1]).cache()
                    # print(best_param_dict[type_of_model]["rank"])
                    result_df = model.transform(train_part_df, validate_part_df, best_param_dict[type_of_model]["rank"])
                    # result_df.show()
                elif type_of_model == "nbcf":
                    result_df = model.transform(train_df.unionAll(validation_df).unionAll(test_input_output[0]),
                                                test_input_output[1], best_param_dict[type_of_model]["rank"])
                else:
                    result_df = model.transform(test_input_output[0], test_input_output[1].drop("TKET")) \
                        .join(test_input_output[1], ["MASV1", "F_MAMH_index"])
                # result_df.show()
                errors = evaluate(result_df, evaluators)
                error_string = faculty + "," + type_of_model + "," + ",".join(
                    str(error) for error in errors)
                print(error_string)
