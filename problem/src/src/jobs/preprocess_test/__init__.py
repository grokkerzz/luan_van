from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, DoubleType, StructType, StructField, StringType, LongType
from module.preprocessor import FullCourseMappingEstimator, FilterCountTransformer, FilterDuplicateUserItemGetMaxTransformer
import glob
from pyspark.sql.functions import col
from sklearn.model_selection import train_test_split
import numpy as np
from pyspark.sql.functions import round, rand, col, sum as spark_sum, collect_list, udf, array, lit, exp, broadcast, count, when


def analyze(spark):
    # allSubDir = glob.glob("data/preprocess_test")
    # allcsv = []
    # for subdir in allSubDir:
    #     files = glob.glob(subdir + "*.csv")
    #     allcsv = allcsv + files
    input_file = "data/df.csv"
    mapping_file = "data/preprocess_test/mapping/mapping.csv"
    mapping_df = spark.read \
        .option("header", "true") \
        .option("treatEmptyValuesAsNulls", "true") \
        .option("inferSchema", "true") \
        .option("charset", "UTF-8") \
        .csv(mapping_file)

    fix_mapping_df = mapping_df\
        .withColumn("F_MAMH", when(col("F_MAMH").cast(IntegerType()).isNotNull(), col("F_MAMH") + lit(".0")).otherwise(col("F_MAMH")))\
        .withColumn("F_MAMH_new", when(col("F_MAMH_new").cast(IntegerType()).isNotNull(), col("F_MAMH_new").cast(IntegerType()) + lit(".0")).otherwise(col("F_MAMH_new")))


    # df = spark.read.format("com.crealytics.spark.excel").option("location", input_file) \
    #     .option("useHeader", "True") \
    #     .option("treatEmptyValuesAsNulls", "true") \
    #     .option("inferSchema", "False") \
    #     .option("addColorColumns", "False") \
    #     .load()  # original input file
    df = spark.read \
        .option("header", "true") \
        .option("treatEmptyValuesAsNulls", "true") \
        .option("inferSchema", "true") \
        .option("charset", "UTF-8") \
        .csv(input_file)

    df = df.select("MASV1", "F_MAMH", "F_MAKH", "TKET", "F_NIENKHOA")
    # df = df.groupBy("MASV1", "F_MAMH", "F_MAKH").agg(collect_list("TKET").alias("list_TKET")).withColumn("TKET", col("list_TKET")[0])
    # df = df.filter(df["F_MAKH"] == "MT")
    # print(df.count())
    # df = df.withColumn("MASV1", df["MASV1"].cast(DoubleType()))
    # df = df.withColumn("MASV1", df["MASV1"].cast(IntegerType()))
    # df = df.withColumn("TKET", df["TKET"].cast(DoubleType()))
    # df = df.groupBy("MASV1", "F_MAMH", "F_MAKH").agg(collect_list("TKET").alias("list_TKET"))\
    #     .withColumn("TKET", col("list_TKET")[0]).drop("list_TKET")
    print("Original df count: {}".format(str(df.count())))
    print("Original df distinct SV_MH distinct count: {}".format(str(df.select("MASV1", "F_MAMH", "F_MAKH").distinct().count())))
    print("Original df distinct F_MAMH count: {}".format(str(df.select("F_MAMH").distinct().count())))
    print("Original df distinct F_MAKH count: {}".format(str(df.select("F_MAKH").distinct().count())))
    course_mapping = FullCourseMappingEstimator()\
        .setItemCol("F_MAMH")\
        .setOutputCol("F_MAMH_new")\
        .fit(fix_mapping_df)

    course_filter = FilterCountTransformer(limit=50)\
        .setItemCol("F_MAMH")

    faculty_filter = FilterCountTransformer(limit=500)\
        .setItemCol("F_MAKH")

    get_max_filter = FilterDuplicateUserItemGetMaxTransformer()\
        .setUserCol("MASV1")\
        .setItemCol("F_MAMH")\
        .setValueCol("TKET")

    mapping_output_df = course_mapping.transform(df).withColumn("F_MAMH", col("F_MAMH_new")).select("MASV1", "F_MAMH", "F_MAKH", "TKET", "F_NIENKHOA")
    mapping_output_df = spark.createDataFrame(mapping_output_df.rdd, mapping_output_df.schema)

    remove_duplicate_df = get_max_filter.transform(mapping_output_df)

    course_filter_output_df = course_filter.transform(remove_duplicate_df)
    # distinct_mapping_count_df = mapping_output_df.select("MASV1", "F_MAMH", "F_MAKH").distinct().groupBy("F_MAMH").agg(count(lit(1)).alias("count_distinct"))
    # mapping_count_df = mapping_output_df.groupBy("F_MAMH").agg(count(lit(1)).alias("count"))
    # list_MMH = distinct_mapping_count_df.join(mapping_count_df, ["F_MAMH"]).withColumn("chenhlech", col("count") - col("count_distinct"))\
    #     .filter(col("chenhlech") != 0)\
    #     .select("F_MAMH") \
    #     .rdd.flatMap(lambda x: x).collect()

    # print(list_MMH)
    # print(len(list_MMH))
    faculty_filter_output_df = faculty_filter.transform(course_filter_output_df)


    print("After mapping MH count: {}".format(str(mapping_output_df.count())))
    print("After mapping SV_MH distinct count: {}".format(str(mapping_output_df.select("MASV1", "F_MAMH", "F_MAKH").distinct().count())))
    print("After remove duplicate MH count: {}".format(str(remove_duplicate_df.count())))
    print("After remove duplicate SV_MH distinct count: {}".format(str(remove_duplicate_df.select("MASV1", "F_MAMH", "F_MAKH").distinct().count())))
    print("After filter MH < 50 data F_MAMH distinct count: {}".format(str(course_filter_output_df.select("F_MAMH").distinct().count())))
    print("After filter MH < 50 data SV_MH distinct count: {}".format(str(course_filter_output_df.select("MASV1", "F_MAMH", "F_MAKH").distinct().count())))
    print("After filter KH < 500 data F_MAKH distinct count: {}".format(str(faculty_filter_output_df.select("F_MAKH").distinct().count())))
    print("After filter KH < 500 data SV_MH distinct count: {}".format(str(faculty_filter_output_df.select("MASV1", "F_MAMH", "F_MAKH").distinct().count())))
    faculty_filter_output_df = faculty_filter_output_df.filter(col("F_NIENKHOA") >= 14)\
        .select("MASV1", "F_MAMH", "F_MAKH", "TKET")
    # # split major
    list_faculty = faculty_filter_output_df.select("F_MAKH").distinct().rdd.flatMap(lambda x: x).collect()
    output_path = "preprocess_output_namhoc"
    print(list_faculty)
    for faculty in list_faculty:
        course_filter_faculty = FilterCountTransformer(limit=15) \
            .setItemCol("F_MAMH")
        faculty_filter_df = course_filter_faculty.transform(faculty_filter_output_df.filter(col("F_MAKH") == faculty))
        # print(faculty_filter_df.count())
        faculty_filter_df = spark.createDataFrame(faculty_filter_df.rdd, faculty_filter_df.schema)
        list_user = faculty_filter_df.select("MASV1").distinct().rdd.flatMap(lambda x: x).collect()
        # print(len(list_user))

        train, validation = train_test_split(np.array(list_user), test_size=0.2, random_state=1)
        train, test = train_test_split(np.array(train), test_size=0.25, random_state=1)
        # print(len(train))
        # print(len(validation))
        # print(len(test))

        user_schema = StructType([StructField("MASV1", LongType())])

        train_data_df = faculty_filter_df.join(spark.createDataFrame([[x] for x in train.tolist()], schema=user_schema), ["MASV1"])
        validation_data_df = faculty_filter_df.join(spark.createDataFrame([[x] for x in validation.tolist()], schema=user_schema), ["MASV1"])
        test_data_df = faculty_filter_df.join(spark.createDataFrame([[x] for x in test.tolist()], schema=user_schema), ["MASV1"])

        # print(train_data_df.count())
        # print(validation_data_df.count())
        # print(test_data_df.count())

        # train_data_df.show()
        # print("new")
        train_data_df.coalesce(1).write.option("header", "true").option("charset", "UTF-8").csv("{}/{}/{}".format(output_path, faculty, "train"))
        validation_data_df.coalesce(1).write.option("header", "true").option("charset", "UTF-8").csv("{}/{}/{}".format(output_path, faculty, "validation"))
        test_data_df.coalesce(1).write.option("header", "true").option("charset", "UTF-8").csv("{}/{}/{}".format(output_path, faculty, "test"))
