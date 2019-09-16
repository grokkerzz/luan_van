from pyspark.sql.types import IntegerType, DoubleType, StructType, StructField, StringType, ArrayType
from pyspark.sql.functions import udf, first, expr, concat, col, count, lit, avg, mean as _mean, array,collect_list, struct
from pyspark.ml.feature import StringIndexer
from pyspark.ml.linalg import Vectors, DenseVector, SparseVector, VectorUDT
import numpy as np
import pyspark.sql.functions as spark_func
from pyspark.sql.window import Window
from pyspark.ml.pipeline import Transformer, Estimator
import time
from pyspark.ml.feature import StringIndexer
import math
from pyspark.ml.evaluation import RegressionEvaluator
import pickle
import glob
import os

# function to calculate cosine similarity between two array
def cosine_similarity(vector1, vector2):
    # vector1 = SparseVector(size, item1)
    # vector2 = SparseVector(size, item2)
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

def sparse_array_to_full_array(sparse_array, size):
     return SparseVector(size, sparse_array).toArray().tolist()

class IBCFTransformer(Transformer):

    def __init__(self, spark, user_col, item_col, item_index_col, grade_col, prediction_col, similarity_df, index_df, rank):
        self.user_col = user_col
        self.item_col = item_col
        self.item_index_col = item_index_col
        self.grade_col = grade_col
        self.prediction_col = prediction_col
        self.item_index_col_2 = self.item_index_col + "_2"
        self.spark = spark
        self.similarity_df = similarity_df
        self.index_df = index_df
        self.rank = rank

    #eliminate item that not exist in train
    def remove_unknown_item(self, train_df, test_df):
        item_df = train_df.select(self.item_col).distinct()
        test_df =  test_df.join(item_df, [self.item_col])
        return test_df

    def _transform(self, dataset):
        pass

    def transform(self, input_df, predict_df, rank=None):

        if rank is None:
            ibcf_rank = self.rank
        else:
            ibcf_rank = rank
        predict_df = predict_df.join(self.index_df, [self.item_col]) \
            .withColumnRenamed(self.item_index_col, self.item_index_col_2)
        input_df = input_df.join(self.index_df, [self.item_col]).drop(self.item_col)

        df = predict_df.join(self.similarity_df, [self.item_index_col_2])\
            .drop(self.grade_col)\
            .join(input_df, [self.user_col, self.item_index_col])\
            .select(self.user_col, self.item_index_col, self.item_index_col_2, "similarity", self.grade_col)

        #print("Data frame for transform")
        #df.show()
        # print(df.count())
        def predict_score(list_score, list_similarity):
            sum_simi = sum(list_similarity)
            if sum_simi == 0:
                return 0.0
            return sum([list_score[i] * list_similarity[i] for i in range(len(list_score))]) / sum_simi

        predict_udf = udf(predict_score, DoubleType())
        window = Window.partitionBy(
                [spark_func.col(self.user_col), spark_func.col(self.item_index_col_2)]).orderBy(
                spark_func.col('similarity').desc())

        result_df = df.select("*", spark_func.rank().over(window).alias("rank")) \
                .filter(spark_func.col("rank") <= ibcf_rank).groupby(self.user_col, self.item_index_col_2) \
                .agg(spark_func.collect_list(self.grade_col).alias("list_score"),
                     spark_func.collect_list("similarity").alias("list_similarity"))
        result_df = result_df.withColumn(self.prediction_col, predict_udf(col("list_score"), col("list_similarity")))

        result_df = result_df.select(self.user_col, self.item_index_col_2, self.prediction_col)
        #predict_df.printSchema()
        #result_df.printSchema()
        result_df = result_df.join(self.spark.createDataFrame(predict_df.rdd, predict_df.schema), [self.user_col, self.item_index_col_2])\
            .drop(self.item_index_col_2)\
            .cache()
        result_df.count()
        #print("Result")
        # print(result_df.count())
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
            self.similarity_df.coalesce(1).write\
                    .option("header", "true")\
                    .option("charset", "UTF-8")\
                    .csv(path+"/similarity")
            self.index_df.coalesce(1).write\
                    .option("header", "true")\
                    .option("charset", "UTF-8")\
                    .csv(path+"/index")        


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
        # df.show(100)
        return IBCFTransformer(spark, config_dictionary["user_col"], config_dictionary["item_col"], config_dictionary["item_index_col"], config_dictionary["grade_col"], config_dictionary["prediction_col"], similarity_df, index_df, config_dictionary["rank"])    


class IBCFEstimator(Estimator):

    def __init__(self, spark, user_col, item_col, item_index_col, grade_col, prediction_col, rank=5):
        self.user_col = user_col
        self.item_col = item_col
        self.item_index_col = item_index_col
        self.grade_col = grade_col
        self.prediction_col = prediction_col
        self.item_index_col_2 = self.item_index_col + "_2"
        self.spark = spark
        self.sparse_array_to_full_array_udf = udf(sparse_array_to_full_array, ArrayType(DoubleType()))
        self.cosine_similarity_udf = udf(cosine_similarity, DoubleType())
        self.to_sparse_vector_udf = udf(to_sparse_vector, VectorUDT())
        self.rank = rank


    #normalize grades
    def normalize_grade(self, df):
        print("Normalizing data ...")
        mean_df = df.groupBy(self.item_col).agg(_mean(self.grade_col).alias("mean"))
        #mean_df.show(100)
        df = df.join(mean_df, [self.item_col])
        #df.show(100)
        df = df.withColumn(self.grade_col, col(self.grade_col) - col("mean"))
        #df.show(100)
        return df, mean_df

    def normalize_result(self, df, mean_df):
        print("Normalizing result...")
        df = df.join(mean_df, [self.item_col])
        # df.show(200)
        #df = df.withColumn(grade_col, col(grade_col) + col("mean"))
        df = df.withColumn(self.prediction_col, col(self.prediction_col) + col("mean")).select(self.user_col, self.item_col, self.grade_col, self.prediction_col)
        # df.show(200)
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


    #eliminate item that not exist in train
    def remove_unknown_item(self, train_df, test_df):
        item_df = train_df.select(self.item_col).distinct()
        test_df =  test_df.join(item_df, [self.item_col])
        return test_df


    #func to transform sparse vector to dense vector
    def sparse_to_array(self, v):
        # print("Coverting featues to dense vector ...")
        v = DenseVector(v)
        new_array = list([float(x) for x in v])
        return new_array

    def _fit(self, transformed_df):
        pass

    def fit(self, transformed_df):
        # transformed_df.show()
        # print("Training ...")
        # print("fit count:{}".format(transformed_df.count()))
        # ratings_pivot = transformed_df.groupBy(self.user_col)\
        #                 .pivot(self.item_index_col)\
        #                 .agg(expr("coalesce(first({}),0.0)".format(self.grade_col))\
        #                 .cast("double"))
        # ratings_pivot.cache()
        # final_df = ratings_pivot.na.fill(0)
        # #final_df.show(1)
        # cols = final_df.columns[1:]

        # schema = StructType([
        #     StructField(self.item_index_col, IntegerType(), True),
        #     StructField("features1", ArrayType(DoubleType()), True)])
        #
        # rows = final_df.select(cols).collect()
        # start = time.time()
        # data_list = [(int(column), [i[column] for i in rows]) for column in cols]
        # end = time.time()
        # elapse = end - start
        # print("Time: " + str(elapse))

        user_index_col = self.user_col + "_index"
        number_of_user = transformed_df.select(self.user_col).distinct().count()
        indexer_model = StringIndexer().setInputCol(self.user_col).setOutputCol(user_index_col)\
            .fit(transformed_df.select(self.user_col).distinct())
        # print(indexer_model.transform(transformed_df).select(user_index_col, self.item_index_col).distinct().count())
        # print(transformed_df.count())
        final_df = indexer_model.transform(transformed_df)\
            .withColumn(user_index_col, col(user_index_col).cast(IntegerType()))\
            .withColumn("user_rating", struct(col(user_index_col), col(self.grade_col)))\
            .groupBy(self.item_index_col)\
            .agg(collect_list("user_rating").alias("features1"))\
            .withColumn("features1", self.to_sparse_vector_udf(col("features1"), lit(number_of_user)))\
            .select(self.item_index_col, "features1")\
            .cache()
        # final_df.show()
        # final_df.printSchema()
        final_df.count()
        copy_df = self.spark.createDataFrame(final_df.rdd, final_df.schema)\
            .withColumnRenamed(self.item_index_col, self.item_index_col_2)\
            .withColumnRenamed("features1", "features2")
        item_similarity = final_df.crossJoin(copy_df)\
            .filter(col(self.item_index_col) != col(self.item_index_col_2))\
            .withColumn("similarity", self.cosine_similarity_udf(col("features1"), col("features2")))\
            .select(self.item_index_col, self.item_index_col_2, "similarity")\
            .cache()
        # final_df.createOrReplaceTempView("FeatureTable")
        # print(final_df.count())
        # item_similarity = self.spark.sql(
        #     "SELECT I1.id as id1, I2.id as id2, I1.features as features1, I2.features as features2  FROM FeatureTable I1, FeatureTable I2 WHERE I1.id != I2.id")
        #
        # cosine_similarity_udf = udf(cosine_similarity, DoubleType())
        # # copy_df = final_df.withColumnRenamed(self.item_index_col,self.item_index_col_2).withColumnRenamed("features1","features2")
        # # item_similarity = final_df.join(copy_df, final_df[self.item_index_col] != copy_df[self.item_index_col_2])
        # item_similarity = item_similarity.withColumn("similarity", cosine_similarity_udf(col("features1"),col("features2"))) \
        #     .withColumnRenamed("id1", self.item_index_col)\
        #     .withColumnRenamed("id2", self.item_index_col_2)\
        #     .select(self.item_index_col, self.item_index_col_2, "similarity")\
        #     .cache()

        # data_list = []
        # # # print("In  for loop")
        # for column in cols:
        #     # print(column)
        #     col_value = [i[column] for i in rows]
        #     #print(col_value)
        #     data = (int(column), col_value)
        #     data_list.append(data)
        #
        #
        # final_df = self.spark.createDataFrame(data_list, schema)
        # print(final_df.count())
        #
        # #final_df.coalesce(1).write.csv("sparse_vector", header=True)
        # # final_df.createOrReplaceTempView("FeatureTable")
        # # item_similarity = self.spark.sql(
        # #     "SELECT I1.id as id1, I2.id as id2, I1.features as features1, I2.features as features2  FROM ItemFactor I1, ItemFactor I2 WHERE I1.id != I2.id")
        #
        # copy_df = final_df.withColumnRenamed(self.item_index_col,self.item_index_col_2).withColumnRenamed("features1","features2")
        # item_similarity = final_df.join(copy_df, final_df[self.item_index_col] != copy_df[self.item_index_col_2])
        # item_similarity = item_similarity.withColumn("similarity", cosine_similarity_udf(col("features1"),col("features2"))) \
        #     .select(self.item_index_col, self.item_index_col_2, "similarity")\
        #     .cache()
        # item_similarity.show()
        # item_similarity.show()
        # item_similarity.count()
        index_df = transformed_df.select(self.item_col, self.item_index_col).distinct()
        return IBCFTransformer(self.spark, self.user_col, self.item_col, self.item_index_col, self.grade_col, self.prediction_col, item_similarity, index_df, self.rank)