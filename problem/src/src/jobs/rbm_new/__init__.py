from pyspark.sql.types import LongType, DoubleType, StructField, StructType
from pyspark.sql import SparkSession
from pyspark.sql.functions import round, col, lit, collect_list, when
from module.collaborative_alluser_array_rbm import RBM, ProbabilitySoftMaxToExpectationModel, ValueToBinarySoftMaxModel,ProbabilitySoftMaxGetMaxIndexModel
from pyspark.ml.evaluation import RegressionEvaluator

#
# def get_train_test(df, spark):
#     user_df = df.select("MASV1").distinct()
#     user_list = user_df.rdd.flatMap(lambda x: x).collect()
#     random.shuffle(user_list)
#     train, test = train_test_split(np.array(user_list), test_size=0.2)
#     user_schema = StructType([StructField("MASV1", LongType())])
#     train_df = df.join(spark.createDataFrame([[x] for x in train.tolist()], schema=user_schema), ["MASV1"])
#     test_df = df.join(spark.createDataFrame([[x] for x in test.tolist()], schema=user_schema), ["MASV1"])
#     return train_df, test_df


def filter_MH(train_df, validate_df, test_df):
    list_filter_MH = ["MI1003", "PE1003", "SP1003", "PE1005", "PE1007",
                      "SP1005", "SP1007", "SP1009", "SP1011", "SP1013",
                      "SP1015", "SP1017", "SP1019", "SP1021", "SP1023",
                      "SP1025", "SP1027", "SP3001", "5007", "5007.0",
                      "5008", "5008.0", "5009", "5009.0"]
    for filter_MH_ele in list_filter_MH:
        train_df = train_df.filter(col("F_MAMH") != filter_MH_ele)
        validate_df = validate_df.filter(col("F_MAMH") != filter_MH_ele)
        test_df = test_df.filter(col("F_MAMH") != filter_MH_ele)
    return train_df, validate_df, test_df


def load_csv_file(path, spark):
    return_df = spark.read \
        .option("header", "true") \
        .option("treatEmptyValuesAsNulls", "true") \
        .option("inferSchema", "true") \
        .option("charset", "UTF-8") \
        .csv(path)
    return_df = return_df.withColumn("MASV1", return_df["MASV1"].cast(DoubleType())) \
        .withColumn("TKET", when(col("TKET") < 5.0, 4.0).otherwise(col("TKET")))
    return_df = return_df.withColumn("MASV1", return_df["MASV1"].cast(LongType())) \
        .withColumn("TKET", when(col("TKET") < 5.0, 4.0).otherwise(col("TKET")))
    return_df = return_df.withColumn("TKET", return_df["TKET"].cast(DoubleType())) \
        .withColumn("TKET", when(col("TKET") < 5.0, 4.0).otherwise(col("TKET")))
    return return_df.select("MASV1", "F_MAMH", "TKET").distinct()

def analyze(spark, cd):
    train_path = "dist/data/train_val_test_K14_K17/MT/train.csv"
    test_path = "dist/data/train_val_test_K14_K17/MT/test.csv"
    validation_path = "dist/data/train_val_test_K14_K17/MT/validation.csv"
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

    train_df = load_csv_file(train_path, spark)
    validation_df = load_csv_file(validation_path, spark)
    test_df = load_csv_file(test_path, spark)
    train_df, validation_df, test_df = filter_MH(train_df, validation_df, test_df)
    test_input_output_df = test_df.randomSplit([0.5, 0.5])
    # test_input_output_df = [load_csv_file("data/test_input_df1.csv", spark), load_csv_file("data/test_output_df1.csv", spark)]



    # train_df.show()
    # preprocess input
    # TKET to int (double score)
    print("#####################Double Score To Index SoftMax######################")

    train_input_rbm_df = train_df.unionAll(validation_df).withColumn("TKET", round(col("TKET")).cast(LongType()))\
        .drop("F_MAKH")
    test_input_rbm_df = test_input_output_df[0].withColumn("TKET", round(col("TKET")).cast(LongType()))\
        .drop("F_MAKH")
    test_input_output_df[1] = test_input_output_df[1].filter(col("TKET") > 1)

    # print(train_input_rbm_df.count())
    # print(train_input_rbm_df.select("MASV1", "F_MAMH", "TKET").distinct().count())
    # print(train_input_rbm_df.select("MASV1", "F_MAMH").distinct().count())
    train_input_rbm_df = train_input_rbm_df.groupBy("MASV1", "F_MAMH").agg(collect_list("TKET").alias("list_TKET"))\
        .withColumn("TKET", col("list_TKET")[0])
    # print(train_input_rbm_df.count())
    # print(train_input_rbm_df.select("MASV1", "F_MAMH", "TKET").distinct().count())
    # print(train_input_rbm_df.select("MASV1", "F_MAMH").distinct().count())


    # print(test_input_rbm_df.count())
    # print(test_input_rbm_df.select("MASV1", "F_MAMH", "TKET").distinct().count())
    # print(test_input_rbm_df.select("MASV1", "F_MAMH").distinct().count())
    test_input_rbm_df = test_input_rbm_df.groupBy("MASV1", "F_MAMH").agg(collect_list("TKET").alias("list_TKET"))\
        .withColumn("TKET", col("list_TKET")[0])
    # print(test_input_rbm_df.count())
    # print(test_input_rbm_df.select("MASV1", "F_MAMH", "TKET").distinct().count())
    # print(test_input_rbm_df.select("MASV1", "F_MAMH").distinct().count())
    # train_input_rbm_df = train_input_rbm_df.withColumn("SoftmaxIndex", col("TKET").cast(LongType()))\
    #                         .withColumn("Active", lit(1))
    # train_input_rbm_df.show()

    # train_input_rbm_df.cache()
    # #to softmax
    print("#####################To Binary SoftMax######################")
    value_to_binary_softmax_model = ValueToBinarySoftMaxModel(spark)\
        .setItemCol("F_MAMH")\
        .setOutputCol("Active")\
        .setSoftMaxUnit(11)\
        .setValueCol("TKET")
    train_input_rbm_df = value_to_binary_softmax_model.transform(train_input_rbm_df)
    train_input_rbm_df.printSchema()
    train_input_rbm_df.show()
    test_input_rbm_df = value_to_binary_softmax_model.transform(test_input_rbm_df)
    # test_input_rbm_df.show()


    print("#####################Training phase######################")
    # print(train_input_rbm_df.count())
    #Create RBM Model
    rbm_model = RBM(spark, cd_list=[int(cd)], cd_iter=[1])\
        .setUserCol("MASV1")\
        .setItemCol("F_MAMH")\
        .setSoftMaxUnit(11)\
        .setValueCol("Active")\
        .setLearningRate(0.1)\
        .setNumberOfHiddenNode(50)\
        .setIterNum(10)\
        .fit(train_input_rbm_df.unionAll(test_input_rbm_df))

    print("#####################Predict phase######################")
    #transform output to expectation (Active la probability)
    prob_to_expect_model = ProbabilitySoftMaxToExpectationModel(spark).setUserCol("MASV1")\
        .setItemCol("F_MAMH")\
        .setValueCol("Active")\
        .setOutputCol("prediction_expectation")

    prob_get_max_model = ProbabilitySoftMaxGetMaxIndexModel(spark).setUserCol("MASV1")\
        .setItemCol("F_MAMH")\
        .setValueCol("Active")\
        .setOutputCol("prediction_max")

    #predict
    output_rbm_df = rbm_model.transform(test_input_rbm_df)
    output_rbm_df.show()
    predict_expectation_df = prob_to_expect_model.transform(output_rbm_df)
    predict_expectation_df = prob_get_max_model.transform(predict_expectation_df)\
        .withColumn("prediction_max", col("prediction_max").cast(DoubleType()))
    predict_expectation_df.show()
    predict_test_df = test_input_output_df[1].join(predict_expectation_df, ["MASV1", "F_MAMH"])
    predict_test_df.show()
    #calculate error
    evaluator_expectation = RegressionEvaluator(metricName="rmse", labelCol="TKET",
                                                predictionCol="prediction_expectation")

    evaluator_max = RegressionEvaluator(metricName="rmse", labelCol="TKET",
                                        predictionCol="prediction_max")
    rmse_expectation = evaluator_expectation.evaluate(predict_test_df)
    rmse_max = evaluator_max.evaluate(predict_test_df)

    print("Root-mean-square error expectation = " + str(rmse_expectation))
    print("Root-mean-square error max = " + str(rmse_max))