from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, DoubleType, StructType, StructField, StringType, ArrayType
from pyspark.sql.functions import udf, first, expr, concat, col, count, lit, avg, mean as _mean, struct, collect_list
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.ml.linalg import Vectors, DenseVector, SparseVector, VectorUDT
from pyspark.ml.feature import VectorAssembler
import numpy as np 
import pyspark.sql.functions as spark_func
from pyspark.sql.window import Window
from pyspark.ml.evaluation import RegressionEvaluator
# from module.ibcf import cosine_similarity
from pyspark.ml.pipeline import Transformer, Estimator
from pyspark.ml.feature import StringIndexer
import math
import pickle
import glob
import os

def cosine_similarity(vector1, vector2):
    dot_product = math.sqrt(vector1.dot(vector1))*math.sqrt(vector2.dot(vector2))
    if dot_product == 0:
        return 0.0
    return float(vector1.dot(vector2) / dot_product)

def to_sparse_vector(list_of_tuple, size):
    # list_of_tuple.sort(key=lambda tup: tup[0])
    # index = [item[0] for item in list_of_tuple]
    # value = [item[1] for item in list_of_tuple]
    dict_index_value = dict(list_of_tuple)
    return SparseVector(size, dict_index_value)

class NBCFTransformer(object):

    def __init__(self, spark, user_col, item_col, item_index_col, grade_col, prediction_col, similarity_df, transformed_df, index_df, rank=5):
        self.user_col = user_col
        self.item_col = item_col
        self.grade_col = grade_col
        self.prediction_col = prediction_col
        self.user_col_2 = "MASV2"
        self.item_index_col = item_index_col
        self.spark = spark
        self.similarity_df = similarity_df
        self.index_df = index_df
        self.transformed_df = transformed_df
        self.rank = rank

    #eliminate item that not exist in train
    def remove_unknown_item(self, train_df, test_df):
        item_df = train_df.select(self.item_col).distinct()
        test_df =  test_df.join(item_df, [self.item_col])
        return test_df

    def _transform(self, dataset):
        pass

    def transform(self, predict_df, rank=None):
        if rank is None:
            local_rank = self.rank
        else:
            local_rank = rank
        # print("Predicting ...")
        predict_df = predict_df.join(self.index_df, [self.item_col])
        renamed_df = self.transformed_df.withColumnRenamed(self.user_col, self.user_col_2)
        df = predict_df.join(self.similarity_df, [self.user_col])\
            .select(self.user_col, self.user_col_2, self.item_index_col, "similarity")\
            .join(renamed_df, [self.user_col_2, self.item_index_col])\
            .select(self.user_col, self.user_col_2, self.item_index_col, "similarity", self.grade_col)
        # print("Data frame for transform")
        # df.show()
        df.cache()

        def predict_score(list_score, list_similarity):
            sum_simi = sum(list_similarity)
            if sum_simi == 0:
                return 0.0
            return sum([list_score[i] * list_similarity[i] for i in range(len(list_score))]) / sum(list_similarity)

        predict_udf = udf(predict_score, DoubleType())
        window = Window.partitionBy(
                [spark_func.col(self.user_col), spark_func.col(self.item_index_col)]).orderBy(
                spark_func.col('similarity').desc())

        result_df = df.select("*", spark_func.rank().over(window).alias("rank")) \
                .filter(spark_func.col("rank") <= local_rank).groupby(self.user_col, self.item_index_col) \
                .agg(spark_func.collect_list(self.grade_col).alias("list_score"),
                     spark_func.collect_list("similarity").alias("list_similarity"))
        result_df = result_df.withColumn(self.prediction_col, predict_udf(spark_func.col("list_score"), spark_func.col("list_similarity")))

        result_df = result_df.select(self.user_col, self.item_index_col, self.prediction_col)
        result_df = result_df.join(predict_df, [self.user_col, self.item_index_col])
        # print("Result")
        # result_df.show()
        return result_df

    # def evaluate_error(self, result_df):
    #     print("Evaluating...")
    #     evaluator_rmse = RegressionEvaluator(metricName="rmse", labelCol=self.grade_col,
    #                                          predictionCol=self.prediction_col)
    #     error = evaluator_rmse.evaluate(result_df)
    #     return error    
        
    def save(self, path):
        model_params = {'user_col': self.user_col, 'item_col': self.item_col, 'item_index_col': self.item_index_col, 'grade_col': self.grade_col, 'prediction_col': self.prediction_col, 'rank': self.rank}
        # # Create directory        
        try:  
            os.makedirs(os.getcwd()+ "/"+ path)
        except OSError:  
            print ("Creation of the directory %s failed" % path)
        else:
            with open(path + '/model_params.dat', 'w+') as model_params_file:
                pickle.dump(model_params, model_params_file)
            self.similarity_df.write\
                    .option("header", "true")\
                    .option("charset", "UTF-8")\
                    .csv(path+"/similarity")
            self.index_df.write\
                    .option("header", "true")\
                    .option("charset", "UTF-8")\
                    .csv(path+"/index")
            self.transformed_df.write\
                    .option("header", "true")\
                    .option("charset", "UTF-8")\
                    .csv(path+"/transformed")                 


    @classmethod
    def load(self, spark, path):
        config_dictionary = []
        if len(glob.glob(path + '/model_params.dat')) == 0:
            return
        with open(path + '/model_params.dat', 'rb') as config_dictionary_file:
            config_dictionary = pickle.load(config_dictionary_file)
        # print(config_dictionary["user_col"])
        similarity_df = spark.read \
            .option("header", "true") \
            .option("treatEmptyValuesAsNulls", "true") \
            .option("inferSchema", "true") \
            .option("charset", "UTF-8") \
            .csv(glob.glob(path+"/similarity"+"/*.csv"))
        index_df = spark.read \
            .option("header", "true") \
            .option("treatEmptyValuesAsNulls", "true") \
            .option("inferSchema", "true") \
            .option("charset", "UTF-8") \
            .csv(glob.glob(path+"/index"+"/*.csv"))
        transformed_df = spark.read \
            .option("header", "true") \
            .option("treatEmptyValuesAsNulls", "true") \
            .option("inferSchema", "true") \
            .option("charset", "UTF-8") \
            .csv(glob.glob(path+"/transformed"+"/*.csv"))          
        # df.show(100)
        return NBCFTransformer(spark, config_dictionary["user_col"], config_dictionary["item_col"], config_dictionary["item_index_col"], config_dictionary["grade_col"], config_dictionary["prediction_col"], similarity_df, transformed_df, index_df, config_dictionary["rank"])    


class NBCFEstimator(Estimator):

    def __init__(self, spark, user_col, item_col, grade_col, prediction_col, rank = 5):
        self.user_col = user_col
        self.item_col = item_col
        self.grade_col = grade_col
        self.prediction_col = prediction_col
        self.user_col_2 = "MASV2"
        self.item_index_col = self.item_col + "_index"
        self.spark = spark
        self.cosine_similarity_udf = udf(cosine_similarity, DoubleType())
        self.to_sparse_vector_udf = udf(to_sparse_vector, VectorUDT())
        self.rank = rank

    #normalize grades
    def normalize_grade(self, df):
        # print("Normalizing data ...")
        mean_df = df.groupBy(self.user_col).agg(_mean(self.grade_col).alias("mean"))
        mean_df_rename = mean_df.withColumnRenamed(self.user_col, "USER")
        #mean_df.show(100)
        df = df.join(mean_df_rename, df[self.user_col] == mean_df_rename["USER"]).drop("USER")
        #df.show(100)
        df = df.withColumn(self.grade_col, col(self.grade_col) - col("mean"))
        #df.show(100)
        return df, mean_df_rename

    def normalize_result(self, df, mean_df):
        # print("Normalizing result...")
        df = df.join(mean_df, df[self.user_col] == mean_df["USER"]).drop("USER")
        df = df.withColumn(self.grade_col, col(self.grade_col) + col("mean"))
        df = df.withColumn(self.prediction_col, col(self.prediction_col) + col("mean")).select(self.user_col, self.item_col, self.grade_col, self.prediction_col)
        #df.show(200)
        return df

    #transform subject code from string to integer
    def create_indexer_df(self, df):
        # print("Creating indexer ...")
        indexer = StringIndexer().setInputCol(self.item_col).setOutputCol(self.item_index_col)
        item_df = df.select(self.item_col).distinct()
        indexer_model = indexer.fit(item_df)
        index_df = indexer_model.transform(item_df)
        # index_df.show()
        return index_df

    #transform subject code from string to integer
    def index_item(self, df, indexer_df):
        # print("Indexing data ...")
        df = df.join(indexer_df, [self.item_col])
        return df.withColumn(self.item_index_col, df[self.item_index_col].cast(IntegerType()))

    #function to transform sparse vector to dense vector
    def sparse_to_array(v):
        # print("Coverting features to dense vector ...")
        v = DenseVector(v)
        new_array = list([float(x) for x in v])
        return new_array

    def _fit(self, transformed_df):
        # print("Training ...")
        
        indexer_df = self.create_indexer_df(transformed_df)
        transformed_df = self.index_item(transformed_df, indexer_df)

        internal_item_index_col = self.item_index_col + "_index"
        number_of_item = transformed_df.select(self.item_index_col).distinct().count()
        indexer_model = StringIndexer().setInputCol(self.item_index_col).setOutputCol(internal_item_index_col)\
            .fit(transformed_df.select(self.item_index_col).distinct())
        feature_df = indexer_model.transform(transformed_df)\
            .withColumn(internal_item_index_col, col(internal_item_index_col).cast(IntegerType()))\
            .withColumn("item_rating", struct(col(internal_item_index_col), col(self.grade_col)))\
            .groupBy(self.user_col)\
            .agg(collect_list("item_rating").alias("features1"))\
            .withColumn("features1", self.to_sparse_vector_udf(col("features1"), lit(number_of_item)))\
            .select(self.user_col, "features1")\
            .cache()
        feature_df.count()
        copy_feature_df_df = self.spark.createDataFrame(feature_df.rdd, feature_df.schema)\
            .withColumnRenamed(self.user_col, self.user_col_2)\
            .withColumnRenamed("features1", "features2")

        window = Window.partitionBy(
                [spark_func.col(self.user_col)]).orderBy(
                spark_func.col('similarity').desc())

        user_similarity = feature_df.crossJoin(copy_feature_df_df)\
            .filter(col(self.user_col) != col(self.user_col_2))\
            .withColumn("similarity", self.cosine_similarity_udf(col("features1"), col("features2")))\
            .select("*", spark_func.rank().over(window).alias("rank")) \
            .filter(spark_func.col("rank") <= 30)\
            .select(self.user_col, self.user_col_2, "similarity")\
            .cache()
        #
        # ratings_pivot = transformed_df.groupBy(self.user_col)\
        #                 .pivot(self.item_index_col)\
        #                 .agg(expr("coalesce(first({}),0.0)".format(self.grade_col))\
        #                 .cast("double"))
        # ratings_pivot.cache()
        # final_df = ratings_pivot.na.fill(0)
        # # final_df.show(1)
        # cols = final_df.columns[1:]
        # assembler = VectorAssembler(inputCols = cols, outputCol = "features1")
        # final_df = assembler.transform(final_df)
        # final_df = final_df.select(self.user_col, "features1")
        # # sparse_to_array_udf = udf(self.sparse_to_array, ArrayType(DoubleType()))
        # # final_df = final_df.withColumn("features1", sparse_to_array_udf("features1"))
        #  # final_df = final_df.withColumn("features", [col(x) for x in final_df.columns[1:]])
        # # final_df.show()
        # final_df.count()
        #
        # cosine_similarity_udf = udf(cosine_similarity, DoubleType())
        # copy_df = final_df.withColumnRenamed(self.user_col,self.user_col_2).withColumnRenamed("features1","features2")
        # user_similarity = final_df.join(copy_df, final_df[self.user_col] != copy_df[self.user_col_2])
        # user_similarity = user_similarity.withColumn("similarity",cosine_similarity_udf(col("features1"),col("features2"))) \
        #     .select(self.user_col, self.user_col_2, "similarity")\
        #     .cache()
        # user_similarity.show()
        index_df = transformed_df.select(self.item_col, self.item_index_col).distinct()
        return NBCFTransformer(self.spark, self.user_col, self.item_col, self.item_index_col, self.grade_col, self.prediction_col, user_similarity, transformed_df, index_df, self.rank)   