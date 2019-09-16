from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.sql.types import IntegerType, DoubleType, StructType, StructField, StringType, ArrayType
from pyspark.sql.functions import collect_set, explode, udf, col, count, lit, avg, mean as _mean
import numpy as np
from pyspark.sql import DataFrame
import pyspark.sql.functions as spark_func
from pyspark.sql.window import Window
import glob
from pyspark.ml.fpm import FPGrowth
from pyspark.ml.pipeline import Transformer, Estimator
import os
import pickle

class FPGEstimator(Estimator):
    def __init__(self, spark, user_col, item_col, grade_col, min_support, min_confidence):
        self.spark = spark
        self.item_col = item_col
        self.user_col = user_col
        self.grade_col = grade_col
        self.list_item_col = self.item_col + "_list"
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.model = FPGrowth(itemsCol=self.list_item_col, minSupport=self.min_support, minConfidence=self.min_confidence, numPartitions=1000)

    def _fit(self, transformed_df):
        train_fp_data = transformed_df.groupBy(self.user_col).agg(collect_set(self.item_col).alias(self.list_item_col)).select(self.user_col,self.list_item_col)
        # train_fp_data = train_fp_data.cache()
        fp_model = self.model.fit(train_fp_data)
        return FPGTransformer(self.spark, self.user_col, self.item_col, self.list_item_col, self.grade_col, fp_model.associationRules)


class FPGTransformer(object):
    def __init__(self, spark, user_col, item_col, list_item_col, grade_col, assoc_df):
        self.spark = spark
        self.user_col = user_col
        self.item_col = item_col
        self.list_item_col = list_item_col
        self.grade_col = grade_col
        self.associationRules = assoc_df
        self.prediction_col = "consequent"

    def transform(self, transformed_df):
        user = [i.MASV1 for i in transformed_df.select(self.user_col).distinct().collect()][0]
        # print(user)
        transform_fp_data = transformed_df.groupBy(self.user_col).agg(collect_set(self.item_col).alias(self.list_item_col)).select(self.user_col,self.list_item_col)
        # transform_fp_data.cache()
        listMH = [i.F_MAMH for i in transformed_df.select(self.item_col).distinct().collect()] 
        def isSubset(arr1):
            for i in range(0, len(arr1)):
                if arr1[i] not in listMH:
                    return 0
            return 1        

        #Testing
        #antecedent|consequent|confidence
         
        tudf = udf(isSubset, IntegerType())
        result_df = self.associationRules.withColumn("suggest", tudf(col("antecedent")))
        result_df = result_df.filter(col("suggest") == 1)               
        result_df = result_df.withColumn(self.item_col, explode(col(self.prediction_col))).drop(self.prediction_col).withColumn(self.user_col,lit(user)).withColumn(self.grade_col,lit(0.0)).select(self.user_col, self.item_col, self.grade_col).distinct()
        suggest_df = result_df.select(self.item_col).subtract(transformed_df.select(self.item_col).distinct())
        result_df = result_df.join(suggest_df, [self.item_col]).distinct()
        return result_df

    def save(self, path):
        model_params = {'user_col': self.user_col, 'item_col': self.item_col, 'list_item_col': self.list_item_col, 'grade_col': self.grade_col}
        # # Create directory        
        try:  
            os.makedirs(os.getcwd()+ "/"+ path)
        except OSError:  
            print ("Creation of the directory %s failed" % path)
        else:
            with open(path + '/model_params.dat', 'w+') as model_params_file:
                pickle.dump(model_params, model_params_file)
            def array_to_string(my_list):
                return "" + ','.join([str(elem) for elem in my_list])

            array_to_string_udf = udf(array_to_string,StringType())

            assoc_df = self.associationRules.withColumn("antecedent_string",array_to_string_udf(col("antecedent"))).drop("antecedent")
            assoc_df = assoc_df.withColumn("consequent_string",array_to_string_udf(col("consequent"))).drop("consequent")
            assoc_df.write\
                    .option("header", "true")\
                    .option("charset", "UTF-8")\
                    .csv(path+"/associationRules")

    @classmethod
    def load(self, spark, path):
        config_dictionary = []
        if len(glob.glob(path + '/model_params.dat')) == 0:
            return
        with open(path + '/model_params.dat', 'rb') as config_dictionary_file:
            config_dictionary = pickle.load(config_dictionary_file)
        # print(config_dictionary["user_col"])
        assoc_df = spark.read \
            .option("header", "true") \
            .option("treatEmptyValuesAsNulls", "true") \
            .option("inferSchema", "true") \
            .option("charset", "UTF-8") \
            .csv(glob.glob(path+"/associationRules"+"/*.csv"))

        def string_to_array(my_list):
                return my_list.split(',')

        string_to_array_udf = udf(string_to_array,ArrayType(StringType()))
        assoc_df = assoc_df.withColumn("antecedent",string_to_array_udf(col("antecedent_string"))).drop("antecedent_string")
        assoc_df = assoc_df.withColumn("consequent",string_to_array_udf(col("consequent_string"))).drop("consequent_string")
        return FPGTransformer(spark, config_dictionary['user_col'], config_dictionary['item_col'], config_dictionary['list_item_col'], config_dictionary['grade_col'], assoc_df)              
                      

