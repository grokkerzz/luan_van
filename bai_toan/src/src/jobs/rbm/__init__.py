from sklearn.model_selection import train_test_split
import numpy as np
import random
from pyspark.sql.types import LongType, DoubleType, StructField, StructType
from pyspark.sql import SparkSession
from pyspark.sql.functions import round, col, lit, collect_list
from module.collaborative_alluser_rbm import RBM, ProbabilitySoftMaxToExpectationModel, ValueToBinarySoftMaxModel
from pyspark.ml.evaluation import RegressionEvaluator


def get_train_test(df, spark):
    user_df = df.select("MASV1").distinct()
    user_list = user_df.rdd.flatMap(lambda x: x).collect()
    random.shuffle(user_list)
    train, test = train_test_split(np.array(user_list), test_size=0.2)
    user_schema = StructType([StructField("MASV1", LongType())])
    train_df = df.join(spark.createDataFrame([[x] for x in train.tolist()], schema=user_schema), ["MASV1"])
    test_df = df.join(spark.createDataFrame([[x] for x in test.tolist()], schema=user_schema), ["MASV1"])
    return train_df, test_df


def load_csv_file(path, spark):
    return_df = spark.read \
        .option("header", "true") \
        .option("treatEmptyValuesAsNulls", "true") \
        .option("inferSchema", "true") \
        .option("charset", "UTF-8") \
        .csv(path)
    return_df = return_df.withColumn("MASV1", return_df["MASV1"].cast(DoubleType()))
    return_df = return_df.withColumn("MASV1", return_df["MASV1"].cast(LongType()))
    return_df = return_df.withColumn("TKET", return_df["TKET"].cast(DoubleType()))
    return return_df.select("MASV1", "F_MAMH", "TKET").distinct()

def analyze(spark):
    # loading input files - pre-processed, load all csv file
    # path = "../data/pre-processed/*.csv"
    # allcsv = glob.glob(path)
    # input_file = allcsv
    # path = "preprocessed_data.csv"
    # allcsv = glob.glob(path)
    # input_file = allcsv
    # create spark session
    # spark = SparkSession.builder.appName("TestRBM").getOrCreate()
    # spark.sparkContext.setCheckpointDir("checkpoint/")
    # spark.sparkContext.setLogLevel("WARN")
    #
    # # read input files
    # df = spark.read \
    #     .option("header", "true") \
    #     .option("treatEmptyValuesAsNulls", "true") \
    #     .option("inferSchema", "true") \
    #     .option("charset", "UTF-8") \
    #     .csv(input_file)
    # df = df.select("MASV1", "F_MAMH", "F_MAKH", "TKET")
    # df = df.filter(df["F_MAKH"] == "MT")
    # # print(df.count())
    # df = df.withColumn("MASV1", df["MASV1"].cast(DoubleType()))
    # df = df.withColumn("MASV1", df["MASV1"].cast(LongType()))
    # df = df.withColumn("TKET", df["TKET"].cast(DoubleType()))
    # df.show()
    # print(df.rdd.getNumPartitions())
    # can tach train va test theo MASV
    spark.sparkContext.setCheckpointDir("hdfs://node3:54311/")
    # spark.sparkContext.setLogLevel("INFO")

    print("#####################Split train test######################")

    # train_df, test_df = get_train_test(df, spark)
    # test_input_output_df = test_df.randomSplit([0.8, 0.2]) #0 la input, 1 la cai output de minh join voi output cua rbm xem ket qua ok ko
    # train_df.coalesce(1).write.csv('train_df.csv')
    # test_input_output_df[0].coalesce(1).write.csv("test_input_df.csv")
    # test_input_output_df[1].coalesce(1).write.csv("test_output_df.csv")
    # train_df.toPandas().to_csv('train_df1.csv')
    # test_input_output_df[0].toPandas().to_csv("test_input_df1.csv")
    # test_input_output_df[1].toPandas().to_csv("test_output_df1.csv")

    train_df = load_csv_file("data/train_df1.csv", spark)
    test_input_output_df = [load_csv_file("data/test_input_df1.csv", spark), load_csv_file("data/test_output_df1.csv", spark)]

    # train_df.show()
    # preprocess input
    # TKET to int (double score)
    print("#####################Double Score To Index SoftMax######################")
    train_input_rbm_df = train_df.withColumn("TKET", round(col("TKET") * 2).cast(LongType()))\
        .drop("F_MAKH")
    test_input_rbm_df = test_input_output_df[0].withColumn("TKET", round(col("TKET") * 2).cast(LongType()))\
        .drop("F_MAKH")

    print(train_input_rbm_df.count())
    print(train_input_rbm_df.select("MASV1", "F_MAMH", "TKET").distinct().count())
    print(train_input_rbm_df.select("MASV1", "F_MAMH").distinct().count())
    train_input_rbm_df = train_input_rbm_df.groupBy("MASV1", "F_MAMH").agg(collect_list("TKET").alias("list_TKET"))\
        .withColumn("TKET", col("list_TKET")[0])
    print(train_input_rbm_df.count())
    print(train_input_rbm_df.select("MASV1", "F_MAMH", "TKET").distinct().count())
    print(train_input_rbm_df.select("MASV1", "F_MAMH").distinct().count())


    print(test_input_rbm_df.count())
    print(test_input_rbm_df.select("MASV1", "F_MAMH", "TKET").distinct().count())
    print(test_input_rbm_df.select("MASV1", "F_MAMH").distinct().count())
    test_input_rbm_df = test_input_rbm_df.groupBy("MASV1", "F_MAMH").agg(collect_list("TKET").alias("list_TKET"))\
        .withColumn("TKET", col("list_TKET")[0])
    print(test_input_rbm_df.count())
    print(test_input_rbm_df.select("MASV1", "F_MAMH", "TKET").distinct().count())
    print(test_input_rbm_df.select("MASV1", "F_MAMH").distinct().count())
    # train_input_rbm_df = train_input_rbm_df.withColumn("SoftmaxIndex", col("TKET").cast(LongType()))\
    #                         .withColumn("Active", lit(1))
    # train_input_rbm_df.show()

    # train_input_rbm_df.cache()
    # #to softmax
    print("#####################To Binary SoftMax######################")
    value_to_binary_softmax_model = ValueToBinarySoftMaxModel(spark)\
        .setItemCol("F_MAMH")\
        .setSoftMaxIndexCol("SoftmaxIndex")\
        .setOutputCol("Active")\
        .setSoftMaxUnit(21)\
        .setValueCol("TKET")
    train_input_rbm_df = value_to_binary_softmax_model.transform(train_input_rbm_df)
    # train_input_rbm_df.show()
    test_input_rbm_df = value_to_binary_softmax_model.transform(test_input_rbm_df)
    # test_input_rbm_df.show()


    print("#####################Training phase######################")
    # print(train_input_rbm_df.count())
    #Create RBM Model
    rbm_model = RBM(spark)\
        .setUserCol("MASV1")\
        .setItemCol("F_MAMH")\
        .setSoftMaxUnit(21)\
        .setSoftMaxIndexCol("SoftmaxIndex")\
        .setValueCol("Active")\
        .setLearningRate(0.1)\
        .setNumberOfHiddenNode(200)\
        .setIterNum(40)\
        .fit(train_input_rbm_df)

    print("#####################Predict phase######################")
    #transform output to expectation (Active la probability)
    prob_to_expect_model = ProbabilitySoftMaxToExpectationModel(spark).setUserCol("MASV1")\
        .setItemCol("F_MAMH")\
        .setSoftMaxIndexCol("SoftmaxIndex")\
        .setValueCol("Active")\
        .setOutputCol("prediction")


    #predict
    output_rbm_df = rbm_model.transform(test_input_rbm_df)
    output_rbm_df.show()
    predict_expectation_df = prob_to_expect_model.transform(output_rbm_df)\
        .withColumn("prediction",  col("prediction") / 2)
    predict_expectation_df.show()
    predict_test_df = test_input_output_df[1].join(predict_expectation_df, ["MASV1", "F_MAMH"])
    predict_test_df.show()
    #calculate error
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="TKET",
                                    predictionCol="prediction")
    rmse = evaluator.evaluate(predict_test_df)
    evaluator = RegressionEvaluator(metricName="mse", labelCol="TKET",
                                    predictionCol="prediction")
    mse = evaluator.evaluate(predict_test_df)
    evaluator = RegressionEvaluator(metricName="mae", labelCol="TKET",
                                    predictionCol="prediction")
    mae = evaluator.evaluate(predict_test_df)
    print("Root-mean-square error = " + str(rmse))
    print("mean-square error = " + str(mse))
    print("mean-absolute error = " + str(mae))
