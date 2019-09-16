from pyspark.sql.types import IntegerType, DoubleType, StructType, StructField, StringType
from pyspark.sql.functions import col, lit, when
from module.preprocessor import FullCourseMappingEstimator

output_user_col = "user"
output_item_col = "items"
output_rating_col = "ratings"


class OutputUtil(object):

    def __init__(self, spark, user_col, item_col, rating_col):
        self.spark = spark
        self.output_folder = "tmp/spark_tmp"
        self.output_template = "tmp/spark_tmp/{}"
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col

    def get_output_path(self, user):
        return self.output_template.format(user)

    def output(self, data, user):
        print(self.get_output_path(user))
        return data.withColumnRenamed(self.user_col, output_user_col)\
            .withColumnRenamed(self.item_col, output_item_col)\
            .withColumnRenamed(self.rating_col, output_rating_col)\
            .select(output_user_col, output_item_col, output_rating_col)\
            .coalesce(1)\
            .write.format('json').save(self.get_output_path(user))


class DataUtil(object):

    def __init__(self, spark):
        self.user_col = "MASV1"
        self.item_col = "F_MAMH"
        self.faculty_col = "F_MAKH"
        self.rating_col = "TKET"
        self.data_dir = "data/train_val_test_1"
        self.train_csv_file = "train.csv"
        self.validation_csv_file = "validation.csv"
        self.test_csv_file = "test.csv"
        self.spark = spark

        self.new_item_col = "F_MAMH_new"
        mapping_file = "data/preprocess_test/mapping/mapping.csv"
        mapping_schema = StructType([
            StructField(self.item_col, StringType(), False),
            StructField(self.new_item_col, StringType(), False)
        ])

        mapping_df = spark.read \
            .option("header", "true") \
            .option("charset", "UTF-8") \
            .schema(mapping_schema)\
            .csv(mapping_file)
        
        fix_mapping_df = mapping_df\
            .withColumn("F_MAMH", when(col("F_MAMH").cast(IntegerType()).isNotNull(), col("F_MAMH") + lit(".0")).otherwise(col("F_MAMH")))\
            .withColumn("F_MAMH_new", when(col("F_MAMH_new").cast(IntegerType()).isNotNull(), col("F_MAMH_new").cast(IntegerType()) + lit(".0")).otherwise(col("F_MAMH_new")))
        
        self.course_mapper = FullCourseMappingEstimator() \
            .setItemCol(self.item_col) \
            .setOutputCol(self.new_item_col) \
            .fit(fix_mapping_df)

    def get_item_df(self, data):
        return data.select(self.item_col).distinct()

    def load_df(self, faculty):
        train_df = self.load_csv("{}/{}/{}".format(self.data_dir, faculty, self.train_csv_file))
        validation_df = self.load_csv("{}/{}/{}".format(self.data_dir, faculty, self.validation_csv_file))
        test_df = self.load_csv("{}/{}/{}".format(self.data_dir, faculty, self.test_csv_file))

        return train_df, validation_df, test_df

    def load_all_df(self, faculty):
        train_df, validation_df, test_df = self.load_df(faculty)
        return train_df.union(validation_df).union(test_df)

    def load_csv(self, path):
        schema = StructType([
            StructField(self.user_col, IntegerType(), False),
            StructField(self.item_col, StringType(), False),
            StructField(self.faculty_col, StringType(), False),
            StructField(self.rating_col, DoubleType(), False),
            StructField("rank", IntegerType(), True)
        ])

        df = self.spark.read \
            .option("header", "true") \
            .option("charset", "UTF-8") \
            .schema(schema)\
            .csv(path)

        df = df.select(self.user_col, self.item_col, self.faculty_col, self.rating_col)
        return df

    def create_df_from_new_data(self, user, items, ratings, faculty):
        item_rating_schema = StructType([
            StructField(self.item_col, StringType(), False), StructField(self.rating_col, DoubleType(), False)
        ])
        return self.spark.createDataFrame(zip(items, ratings), schema=item_rating_schema)\
            .withColumn(self.item_col, when(col(self.item_col).cast(IntegerType()).isNotNull(), col(self.item_col) + lit(".0")).otherwise(col(self.item_col)))\
            .withColumn(self.user_col, lit(int(user)).cast(IntegerType()))\
            .withColumn(self.faculty_col, lit(faculty).cast(StringType()))\
            .select(self.user_col,self.item_col, self.faculty_col,self.rating_col)

    def create_df_from_new_data_without_rating(self, user, items, faculty):
        item_rating_schema = StructType([
            StructField(self.item_col, StringType(), False)
        ])
        return self.spark.createDataFrame(items, schema=item_rating_schema)\
            .withColumn(self.item_col, when(col(self.item_col).cast(IntegerType()).isNotNull(), col(self.item_col) + lit(".0")).otherwise(col(self.item_col)))\
            .withColumn(self.user_col, lit(int(user)).cast(IntegerType()))\
            .withColumn(self.faculty_col, lit(faculty).cast(StringType()))\
            .select(self.user_col,self.item_col, self.faculty_col)

    def mapping_course(self, data):
        return self.course_mapper.transform(data)\
            .drop(self.item_col)\
            .withColumnRenamed(self.new_item_col, self.item_col).distinct()

    def get_item_col(self):
        return self.item_col

    def get_user_col(self):
        return self.user_col

    def get_rating_col(self):
        return self.rating_col
