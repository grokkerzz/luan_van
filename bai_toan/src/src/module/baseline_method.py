from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.sql.types import IntegerType, DoubleType, LongType, StructField, StringType
from pyspark.sql.functions import udf, col, rank, collect_list, avg
import numpy as np
from pyspark.sql import DataFrame
import pyspark.sql.functions as spark_func
from pyspark.sql.window import Window
import glob
from pyspark.ml.pipeline import Transformer, Estimator
from .custom_params import HasUserCol, HasItemCol, HasValueCol, HasOutputCol


class MeanTransformer(HasUserCol, HasItemCol, HasValueCol, HasOutputCol, Transformer):

    def __init__(self, spark):
        super(MeanTransformer, self).__init__()
        self.spark = spark

    def transform(self, input_df, predict_course_df):
        mean_df = input_df.groupBy(self.getUserCol())\
            .agg(avg(col(self.getValueCol())).alias(self.getOutputCol()))\
            .select(self.getUserCol(), self.getOutputCol())
        predict_df = predict_course_df.join(mean_df, [self.getUserCol()])
        return predict_df

    def _transform(self, dataset):
        pass
