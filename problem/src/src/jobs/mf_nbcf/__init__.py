from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.sql.types import IntegerType, DoubleType, StructType, StructField, StringType
from module.mf_nbcf import IBCFEstimator, IBCFWithItemFactor
from pyspark.sql.functions import udf, col, rank, collect_list
import time


def load_csv(spark, path):
    df = spark.read \
        .option("header", "true") \
        .option("treatEmptyValuesAsNulls", "true") \
        .option("inferSchema", "true") \
        .option("charset", "UTF-8") \
        .csv(path)
    df.show()
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


def test(spark, train, test_input, test_output, rank_list):
    evaluator_rmse = RegressionEvaluator(metricName="rmse", labelCol="TKET",
                                         predictionCol="prediction")
    evaluator_mse = RegressionEvaluator(metricName="mse", labelCol="TKET",
                                       predictionCol="prediction")
    evaluator_mae = RegressionEvaluator(metricName="mae", labelCol="TKET",
                                       predictionCol="prediction")
    evaluators = [evaluator_rmse, evaluator_mse, evaluator_mae]
    error_list_als = {}
    error_list_ibcf = {}
    error_list_combine = {}
    for i in range(len(rank_list)):
        als = ALS(rank=rank_list[i], maxIter=15, regParam=0.01, userCol="MASV1",
                       itemCol="F_MAMH_index", ratingCol="TKET", coldStartStrategy="drop", nonnegative=True)
        als_input = train.unionAll(test_input)
        start = time.time()
        als_model = als.fit(als_input)
        predict_als = als_model.transform(test_output)
        predict_als.show()
        end = time.time()
        print("\nExecution of als took %s seconds" % ( end - start))

        # test_input.select("F_MAMH_index").distinct().filter(col("F_MAMH_index") == 0 ).show()
        # als_model.itemFactors.filter(col("id") == 1).show()
        # als_model.itemFactors.filter(col("id") == 10).show()
        # als_model.itemFactors.filter(col("id") == 11).show()

        ibcf_model = IBCFWithItemFactor(spark, als_model.itemFactors)\
            .setUserCol("MASV1")\
            .setItemCol("F_MAMH_index")\
            .setValueCol("TKET")\
            .setRank(5)
        start = time.time()
        predict_ibcf = ibcf_model.transform(test_input,  test_output.drop("TKET"))
        predict_ibcf.show()
        end = time.time()
        print("\nExecution of als_ibcf took %s seconds" % ( end - start))
        
        combine = predict_ibcf.withColumnRenamed("prediction","prediction_ibcf")\
            .join(predict_als.withColumnRenamed("prediction","prediction_als"),["MASV1", "F_MAMH_index"])\
            .withColumn("prediction", (col("prediction_ibcf") + col("prediction_als"))/2)
        combine.show()

        predict_ibcf = predict_ibcf.join(test_output, ["MASV1", "F_MAMH_index"])
        # predict_als.show()
        predict_ibcf.show()
        # predicted = predict_model.predict_using_cosine_similarity(test_input, test_output)

        # predicted  = predicted.withColumn("TKET", col("TKET") + col("mean")).withColumn("prediction", col("prediction") + col("mean"))

        error_als = [evaluator.evaluate(predict_als) for evaluator in evaluators]
        error_ibcf = [evaluator.evaluate(predict_ibcf) for evaluator in evaluators]
        error_combine = [evaluator.evaluate(combine) for evaluator in evaluators]

        error_list_als[rank_list[i]] = error_als
        error_list_ibcf[rank_list[i]] = error_ibcf
        error_list_combine[rank_list[i]] = error_combine

    return error_list_als, error_list_ibcf,error_list_combine


def analyze(spark):
    # allSubDir = glob.glob("data/")
    # allcsv = []
    # for subdir in allSubDir:
    #     files = glob.glob(subdir + "*.csv")
    #     allcsv = allcsv + files
    # input_file = allcsv
    # train_all_data_path = "data/mf_nbcf/train_all_data.csv"
    # train_MT_data_path = "data/mf_nbcf/train_MT_data.csv"
    # test_MT_input_data_path = "data/mf_nbcf/test_MT_input_data.csv"
    # test_MT_output_data_path = "data/mf_nbcf/test_MT_output_data.csv"

    train_path = "data/MT/train.csv"
    test_path = "data/MT/test.csv"
    validation_path = "data/MT/validation.csv"

    spark = SparkSession.builder.appName("BuildModel").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    # df = spark.read.format("com.crealytics.spark.excel").option("location", input_file) \
    #     .option("useHeader", "True") \
    #     .option("treatEmptyValuesAsNulls", "true") \
    #     .option("inferSchema", "False") \
    #     .option("addColorColumns", "False") \
    #     .load()  # original input file
    # train_all_df = load_csv(spark, train_all_data_path)

    train_df = load_csv(spark, train_path)

    validation_df = load_csv(spark, validation_path)

    test_df = load_csv(spark, test_path)

    all_df = train_df.unionAll(validation_df).unionAll(test_df)

    all_item_df = all_df.select("F_MAMH").distinct()
    print(all_item_df.count())

    #index
    item_indexer = StringIndexer().setInputCol("F_MAMH").setOutputCol("F_MAMH_index")
    indexer_model = item_indexer.fit(all_item_df)
    item_index_df = indexer_model.transform(all_item_df)

    train_df = indexer_model.transform(train_df).withColumn("F_MAMH_index", col("F_MAMH_index").cast(IntegerType()))
    test_df = indexer_model.transform(test_df).withColumn("F_MAMH_index", col("F_MAMH_index").cast(IntegerType()))
    validation_df = indexer_model.transform(validation_df).withColumn("F_MAMH_index", col("F_MAMH_index").cast(IntegerType()))

    test_input_output = test_df.randomSplit([0.5, 0.5])

    rank_list_als = [1, 2, 3, 4, 5, 7]

    all_als_model_error, all_ibcf_model_error, all_combine_model_error = test(spark, train_df.unionAll(validation_df), test_input_output[0], test_input_output[1], rank_list_als)
    # MT_model_error = test(spark, train_MT_df, test_MT_input_data, test_MT_output_data, rank_list_als)

    print("model,rank,rmse,mse,mae")
    for key, value in all_als_model_error.items():
        error_string = "all_als" + "," + str(key) + "," + ",".join(str(error) for error in value)
        print(error_string)

    for key, value in all_ibcf_model_error.items():
        error_string = "all_ibcf" + "," + str(key) + "," + ",".join(str(error) for error in value)
        print(error_string)

    for key, value in all_combine_model_error.items():
        error_string = "all_combine" + "," + str(key) + "," + ",".join(str(error) for error in value)
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