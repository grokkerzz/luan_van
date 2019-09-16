from pyspark.ml.pipeline import Estimator, Transformer
from pyspark.sql import DataFrame
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import round, rand, col, sum as spark_sum, collect_list, udf, array, lit, exp, broadcast, count
from .custom_params import HasUserCol, HasItemCol, HasValueCol, HasNumberOfHiddenNode, HasSoftMaxUnit, HasLearningRate, \
    HasOutputCol, HasIterNum
from pyspark.sql.types import LongType, DoubleType, ArrayType
from pyspark import StorageLevel
import random
import numpy as np
import subprocess
import time


def softmax_stable(X, theta = 1.0, axis = None):
    # """
    # Compute the softmax of each element along an axis of X.
    #
    # Parameters
    # ----------
    # X: ND-Array. Probably should be floats.
    # theta (optional): float parameter, used as a multiplier
    #     prior to exponentiation. Default = 1.0
    # axis (optional): axis to compute values along. Default is the
    #     first non-singleton axis.
    #
    # Returns an array the same size as X. The result will sum to 1
    # along the specified axis.
    # """
    #
    # X = np.array(X)
    #
    # # make X at least 2d
    # y = np.atleast_2d(X)
    #
    # # find axis
    # if axis is None:
    #     axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)
    #
    # # multiply y against the theta parameter,
    # y = y * float(theta)
    #
    # # subtract the max for numerical stability
    # y = y - np.expand_dims(np.max(y, axis = axis), axis)
    #
    # # exponentiate y
    # y = np.exp(y)
    #
    # # take the sum along the specified axis
    # ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)
    #
    # # finally: divide elementwise
    # p = y / ax_sum
    #
    # # flatten if X was 1D
    # if len(X.shape) == 1: p = p.flatten()
    #
    # return p.tolist()

    """Compute softmax values for each sets of scores in x."""
    X = np.array(X)
    e_x = np.exp(X - np.max(X))
    result = e_x / e_x.sum(axis=0) # only difference
    return result.tolist()


def array_dot(a, b):
    return np.dot(a, b).tolist()


def array_sum_array(a):
    return np.sum(a, axis=0).tolist()


def array_sum_all(a):
    return np.sum(a).tolist()


def array_multiply(a, b):
    return np.multiply(a, b).tolist()


def random_numpy(mean_gaussian, std, size):
    return np.random.normal(mean_gaussian, std, size).tolist()


def array_add(a,b):
    return np.add(a, b).tolist()


def array_div(a, b):
    return np.true_divide(a, b).tolist()


def array_minus(a,b):
    return np.add(np.array(a), -np.array(b)).tolist()


def array_exp(a):
    return np.exp(a).tolist()


def sigmoid(x):
    array_x = np.array(x)
    return np.where(array_x >= 0,
                    1 / (1 + np.exp(-array_x)),
                    np.exp(array_x) / (1 + np.exp(array_x))).tolist()


def softmax_sampling(list_probability):
    result_array = np.zeros(len(list_probability), dtype=int)
    choosen = np.random.choice(range(len(result_array)), None, False, list_probability)
    result_array[choosen] = 1
    return result_array.tolist()


class RBMCore(HasUserCol, HasItemCol, HasValueCol, HasNumberOfHiddenNode, HasSoftMaxUnit, Transformer):

    #some column name
    col_name_item_index = "i"
    col_name_hidden_node_index = "j"
    col_name_value = "value"
    col_name_value_visible = "value_visible"
    col_name_value_hidden = "value_hidden"
    col_name_value_weight = "value_weight"
    col_name_mul = "weight_mul_value"
    col_name_add = "add_bias_value"
    col_name_probability = "probability"
    col_name_nominator = "nominator"
    col_name_denominator = "denominator"
    col_name_delta_value = "delta_value"
    col_name_old_value = "old_value"
    col_name_value_0 = "value_0"
    col_name_value_k = "value_k"
    col_name_activation = "activation"

    def __init__(self, spark, weight, visible_layer_bias, hidden_layer_bias, item_index_mapping):
        super(RBMCore, self).__init__()
        self._weight = weight
        self._visible_layer_bias = visible_layer_bias
        self._hidden_layer_bias = hidden_layer_bias
        self._item_index_mapping = item_index_mapping
        self.softmax_sampling_udf = udf(softmax_sampling, ArrayType(LongType(), containsNull=False))
        self.sigmoid_double_udf = udf(sigmoid, DoubleType())
        self.sigmoid_array_udf = udf(sigmoid, ArrayType(DoubleType(), containsNull=False))
        self.soft_max_udf = udf(softmax_stable, ArrayType(DoubleType(), containsNull=False))
        self.array_dot_udf = udf(array_dot, DoubleType())
        self.array_multiply_udf = udf(array_multiply, ArrayType(DoubleType(), containsNull=False))
        self.array_sum_array_udf = udf(array_sum_array, ArrayType(DoubleType(), containsNull=False))
        self.array_sum_all_udf = udf(array_sum_all, DoubleType())
        self.array_add_udf = udf(array_add, ArrayType(DoubleType(), containsNull=False))
        self.array_minus_double_udf = udf(array_minus, ArrayType(DoubleType(), containsNull=False))
        self.array_minus_long_udf = udf(array_minus, ArrayType(LongType(), containsNull=False))
        self.array_div_udf = udf(array_div, ArrayType(DoubleType(), containsNull=False))
        self.array_exp_udf = udf(array_exp, ArrayType(DoubleType(), containsNull=False))
        # self.hdfs_path = "hdfs://node3:54311/"
        self.hdfs_path = "hdfs://10.1.1.24:9000/"
        self.spark = spark

    def _transform(self, dataset):
        dataset = dataset.join(self._item_index_mapping, [self.getItemCol()]).drop(RBMCore.col_name_item_index)
        input_df = dataset.join(self._item_index_mapping, [self.getItemCol()])\
            .withColumn(RBMCore.col_name_item_index, col(RBMCore.col_name_item_index).cast(LongType()))\
            .withColumnRenamed(self.getValueCol(), RBMCore.col_name_value) \
            .select(self.getUserCol(), RBMCore.col_name_item_index, RBMCore.col_name_value)
        input_df = self.spark.createDataFrame(input_df.rdd, input_df.schema).cache()
        # input_df.printSchema()
        # input_df.show()
        hidden_df = self.calculate_prob_h_v(input_df).withColumnRenamed(RBMCore.col_name_probability, RBMCore.col_name_value)
        # hidden_df = self.calculate_prob_h_v(input_df)
        # hidden_df.show()
        # hidden_df = self.sample_h(hidden_df)\
        #     .select(self.getUserCol(), RBMCore.col_name_hidden_node_index, RBMCore.col_name_value)
        hidden_df.show()
        visible_df = self.calculate_prob_v_h(hidden_df)
        # visible_df.show()
        return visible_df.withColumnRenamed(RBMCore.col_name_probability, self.getValueCol())\
            .join(self._item_index_mapping, [RBMCore.col_name_item_index])\
            .drop(RBMCore.col_name_item_index)

    def calculate_prob_v_h(self, hidden_activation, user_visible_item=None):
        # h is a dataframe with 3 column (self.getUserCol(), col_name_hidden_node_index, value)
        # value is 0 or 1
        # user visible item is item observe by user useful for training phase to generate specific rbm for user
        # print("begin calculate_prob_v_h")
        if user_visible_item is None:
            hidden_activation_weight_df = hidden_activation.withColumnRenamed(RBMCore.col_name_value, RBMCore.col_name_value_hidden) \
                .join(broadcast(self._weight.withColumnRenamed(RBMCore.col_name_value, RBMCore.col_name_value_weight)),
                [RBMCore.col_name_hidden_node_index])
        else:
            user_rbm_weight = user_visible_item.join(broadcast(self._weight), [RBMCore.col_name_item_index])
            hidden_activation_weight_df = hidden_activation.withColumnRenamed(RBMCore.col_name_value, RBMCore.col_name_value_hidden) \
                .join(user_rbm_weight.withColumnRenamed(RBMCore.col_name_value, RBMCore.col_name_value_weight),
                [self.getUserCol(), RBMCore.col_name_hidden_node_index])
        #hidden_activation_weight_df 4 col (self.getUSerCol(), col_name_hidden_node_index, col_name_item_index, value (array), col_name_value_hidden)
        probability_df = hidden_activation_weight_df\
            .withColumn(RBMCore.col_name_mul, self.array_multiply_udf(col(RBMCore.col_name_value_weight), col(RBMCore.col_name_value_hidden))) \
            .groupBy(self.getUserCol(), RBMCore.col_name_item_index) \
            .agg(collect_list(RBMCore.col_name_mul).alias("sum")) \
            .withColumn("sum", self.array_sum_array_udf(col("sum")))\
            .join(broadcast(self._visible_layer_bias.withColumnRenamed(RBMCore.col_name_value, RBMCore.col_name_value_visible)),
                  [RBMCore.col_name_item_index]) \
            .withColumn(RBMCore.col_name_add, self.array_add_udf(col("sum"), col(RBMCore.col_name_value_visible)))\
            .withColumn(RBMCore.col_name_probability, self.soft_max_udf(col(RBMCore.col_name_add))) \
            .select(self.getUserCol(), RBMCore.col_name_item_index, RBMCore.col_name_probability)
        # print("probability_v_h_df count: {}".format(probability_df.count()))
        # probability_df.show()
        return probability_df

    def sample_v(self, probability_df):
        # print("begin sample_v")
        sample_v_df = probability_df\
            .withColumn(RBMCore.col_name_value, self.softmax_sampling_udf(col(RBMCore.col_name_probability))) \
            .select(self.getUserCol(), RBMCore.col_name_item_index, RBMCore.col_name_probability,
                    RBMCore.col_name_value)
        # print("sample_v_df count: {}".format(sample_v_df.count()))
        # sample_v_df.show()
        # print("done sample_v")
        return sample_v_df

    def calculate_prob_h_v(self, visible_activation):
        # print("begin calculate_prob_h_v")
        # v is a dataframe with 3 column (self.getUserCol(),col_name_item_index, value)
        # for each item index there is only 1 softmax has value of 1 (otherwise 0)
        # print("user count: " + str(visible_activation.select(self.getUserCol()).distinct().count()))
        prob_h_df = visible_activation.withColumnRenamed(RBMCore.col_name_value, RBMCore.col_name_value_visible) \
            .join(broadcast(self._weight.withColumnRenamed(RBMCore.col_name_value, RBMCore.col_name_value_weight)),
                  [RBMCore.col_name_item_index]) \
            .withColumn(RBMCore.col_name_mul, self.array_dot_udf(col(RBMCore.col_name_value_visible), col(RBMCore.col_name_value_weight))) \
            .groupBy(self.getUserCol(), RBMCore.col_name_hidden_node_index) \
            .agg(spark_sum(RBMCore.col_name_mul).alias("sum")) \
            .join(broadcast(self._hidden_layer_bias.withColumnRenamed(RBMCore.col_name_value, RBMCore.col_name_value_hidden)),
                  [RBMCore.col_name_hidden_node_index]) \
            .withColumn(RBMCore.col_name_probability, self.sigmoid_double_udf(col("sum") + col(RBMCore.col_name_value_hidden)))\
            .select(self.getUserCol(), RBMCore.col_name_hidden_node_index, RBMCore.col_name_probability)
        # print("prob_h_v_df count: {}".format(prob_h_df.count()))
        # prob_h_df.show()
        # print("done calculate_prob_h_v")
        return prob_h_df

    def sample_h(self, probability_df):
        # print("begin sample_h")
        sample_h_df = probability_df\
            .withColumn(RBMCore.col_name_value, (rand() < col(RBMCore.col_name_probability)).cast(LongType())) \
            .select(self.getUserCol(), RBMCore.col_name_hidden_node_index, RBMCore.col_name_probability, RBMCore.col_name_value)
        # print("sample_h_df count: {}".format(sample_h_df.count()))
        # sample_h_df.show()
        # print("done sample_h")
        return sample_h_df

    def gibbs_sampling(self, visible_activation, number_of_cycle, use_all_weight=True):
        # print("begin gibbs_sampling")
        if use_all_weight is True:
            user_visible_item_df = None
        else:
            user_visible_item_df = visible_activation.select(self.getUserCol(), RBMCore.col_name_item_index).distinct()

        # print("w,v_b,h_b count: {},{},{}".format(
        #     str(self._weight.count()), str(self._visible_layer_bias.count()), str(self._hidden_layer_bias.count())))

        # ph0_df = self.calculate_prob_h_v(visible_activation)\
        #     .select(self.getUserCol(), RBMCore.col_name_hidden_node_index, RBMCore.col_name_probability)
        # ph0_df.cache()
        ph0_df = None
        vk = visible_activation
        for i in range(number_of_cycle):
            phk = self.calculate_prob_h_v(vk)
            if ph0_df is None:
                ph0_df = phk
                ph0_df = ph0_df.cache()
            hk = self.sample_h(phk)\
                .select(self.getUserCol(), RBMCore.col_name_hidden_node_index, RBMCore.col_name_value)
            # hk.cache()
            vk = self.sample_v(self.calculate_prob_v_h(hk, user_visible_item_df)) \
                .select(self.getUserCol(), RBMCore.col_name_item_index, RBMCore.col_name_value)
            # vk.cache()
            # if i % 2 == 1:
            #     hk = hk.cache()
            #     vk = vk.cache()
            #     hk = hk.checkpoint(True)
            #     vk = vk.checkpoint(True)

            # check number of user item (verify correct rbm use per user)
            # print("original v_df: " + str(visible_activation.count()) + ", after sampling v_df: " + str(vk.count()))

        phk_df = self.calculate_prob_h_v(vk)
        phk_df = phk_df.cache()
        vk_df = self.spark.createDataFrame(vk.rdd, vk.schema).cache() #cloning prevent join bug
        vk = None
        # vk.explain()
        # visible_activation.explain()
        # ph0_df.explain()
        # phk_df.explain()

        # visible_activation.show()
        # vk.show()
        # ph0_df.show()
        # phk_df.show()
        # print("done gibbs_sampling")
        return vk_df, ph0_df, phk_df

    # v0_df and vk_df is dataframe with 3 column(self.getUserCol(),col_name_item_index, col_name_value)
    # ph0_df and phk_df is dataframe with 3 column(self.getUserCol(),col_name_hidden_node_index, col_name_probability)
    def compute_gradient(self, v0_df, vk_df, ph0_df, phk_df, number_of_user):
        # print("begin compute_gradient")
        # gradient of hidden bias
        # print("compute gradient")
        # vk_df = self.spark.createDataFrame(vk_df.rdd, vk_df.schema).cache() #cloning prevent join bug
        dhb_df = ph0_df.withColumnRenamed(RBMCore.col_name_probability, RBMCore.col_name_value_0) \
            .join(phk_df.withColumnRenamed(RBMCore.col_name_probability, RBMCore.col_name_value_k),
                  [self.getUserCol(), RBMCore.col_name_hidden_node_index]) \
            .withColumn(RBMCore.col_name_value, (col(RBMCore.col_name_value_0) - col(RBMCore.col_name_value_k)) / number_of_user) \
            .groupBy(RBMCore.col_name_hidden_node_index) \
            .agg(spark_sum(RBMCore.col_name_value).alias("sum_" + RBMCore.col_name_value)) \
            .withColumnRenamed("sum_" + RBMCore.col_name_value, RBMCore.col_name_value) \
            .select(RBMCore.col_name_hidden_node_index, RBMCore.col_name_value)
        # dhb_df.printSchema()
        # dhb_df.show()
        # gradient of visible bias
        dvb_df = v0_df.withColumnRenamed(RBMCore.col_name_value, RBMCore.col_name_value_0)\
            .join(vk_df.withColumnRenamed(RBMCore.col_name_value, RBMCore.col_name_value_k),
            [self.getUserCol(), RBMCore.col_name_item_index]) \
            .withColumn(RBMCore.col_name_value, self.array_minus_long_udf(col(RBMCore.col_name_value_0), col(RBMCore.col_name_value_k)))\
            .withColumn(RBMCore.col_name_value, self.array_div_udf(col(RBMCore.col_name_value), lit(number_of_user))) \
            .groupBy(RBMCore.col_name_item_index) \
            .agg(collect_list(RBMCore.col_name_value).alias("list_" + RBMCore.col_name_value))\
            .withColumn(RBMCore.col_name_value, self.array_sum_array_udf(col("list_" + RBMCore.col_name_value)))\
            .select(RBMCore.col_name_item_index, RBMCore.col_name_value)
        # dvb_df.printSchema()
        # dvb_df.show()
        # gradient of weight
        weight_0 = v0_df.withColumnRenamed(RBMCore.col_name_value, RBMCore.col_name_activation) \
            .join(ph0_df, [self.getUserCol()]) \
            .withColumn(RBMCore.col_name_value_0, self.array_multiply_udf(col(RBMCore.col_name_activation), col(RBMCore.col_name_probability))) \
            .groupBy(RBMCore.col_name_item_index, RBMCore.col_name_hidden_node_index) \
            .agg(collect_list(RBMCore.col_name_value_0).alias("list_" + RBMCore.col_name_value)) \
            .withColumn(RBMCore.col_name_value_0, self.array_sum_array_udf(col("list_" + RBMCore.col_name_value))) \
            .select(RBMCore.col_name_item_index, RBMCore.col_name_hidden_node_index, RBMCore.col_name_value_0) \
            # .repartition(ncore, [col_name_item_index, col_name_soft_max_unit_index])
        # weight_0.printSchema()
        # weight_0.cache()
        # weight_0.show()
        weight_k = vk_df.withColumnRenamed(RBMCore.col_name_value, RBMCore.col_name_activation) \
            .join(phk_df, [self.getUserCol()]) \
            .withColumn(RBMCore.col_name_value_k, self.array_multiply_udf(col(RBMCore.col_name_activation), col(RBMCore.col_name_probability))) \
            .groupBy(RBMCore.col_name_item_index, RBMCore.col_name_hidden_node_index) \
            .agg(collect_list(RBMCore.col_name_value_k).alias("list_" + RBMCore.col_name_value)) \
            .withColumn(RBMCore.col_name_value_k, self.array_sum_array_udf(col("list_" + RBMCore.col_name_value)))\
            .select(RBMCore.col_name_item_index, RBMCore.col_name_hidden_node_index, RBMCore.col_name_value_k) \
            # .repartition(ncore, [col_name_item_index, col_name_soft_max_unit_index])
        # weight_k.printSchema()
        # weight_k.cache()
        # weight_k.show()
        dw_df = weight_0.join(broadcast(weight_k),
                              [RBMCore.col_name_item_index, RBMCore.col_name_hidden_node_index]) \
            .withColumn(RBMCore.col_name_value, self.array_minus_double_udf(col(RBMCore.col_name_value_0), col(RBMCore.col_name_value_k))) \
            .withColumn(RBMCore.col_name_value,
                        self.array_div_udf(col(RBMCore.col_name_value), lit(number_of_user))) \
            .select(RBMCore.col_name_item_index, RBMCore.col_name_hidden_node_index, RBMCore.col_name_value)
        # dw_df.printSchema()
        # dw_df.show()
        # print("done compute_gradient")
        return dw_df, dvb_df, dhb_df

    def checkPoint(self):
        pass

    def update_parameter(self, dw, db_v, db_h, learning_rate):
        # print("begin update_parameter")
        # print("Before update w,v_b,h_b count: {},{},{}".format(
        #     str(self._weight.count()), str(self._visible_layer_bias.count()), str(self._hidden_layer_bias.count())))
        self._visible_layer_bias = self._visible_layer_bias.withColumnRenamed(RBMCore.col_name_value, RBMCore.col_name_old_value)\
            .join(broadcast(db_v.withColumnRenamed(RBMCore.col_name_value, RBMCore.col_name_delta_value)),
                  [RBMCore.col_name_item_index]) \
            .withColumn(RBMCore.col_name_delta_value, self.array_multiply_udf(col(RBMCore.col_name_delta_value), lit(learning_rate))) \
            .withColumn(RBMCore.col_name_value, self.array_add_udf(col(RBMCore.col_name_old_value), col(RBMCore.col_name_delta_value)))\
            .select(RBMCore.col_name_item_index, RBMCore.col_name_value)
        self._hidden_layer_bias = self._hidden_layer_bias.withColumnRenamed(RBMCore.col_name_value, RBMCore.col_name_old_value)\
            .join(broadcast(db_h.withColumnRenamed(RBMCore.col_name_value, RBMCore.col_name_delta_value)),
                  [RBMCore.col_name_hidden_node_index]) \
            .withColumn(RBMCore.col_name_value,
                        col(RBMCore.col_name_old_value) + (col(RBMCore.col_name_delta_value) * learning_rate)) \
            .select(RBMCore.col_name_hidden_node_index, RBMCore.col_name_value)
        self._weight = self._weight.withColumnRenamed(RBMCore.col_name_value, RBMCore.col_name_old_value)\
            .join(broadcast(dw.withColumnRenamed(RBMCore.col_name_value, RBMCore.col_name_delta_value)),
                  [RBMCore.col_name_item_index, RBMCore.col_name_hidden_node_index]) \
            .withColumn(RBMCore.col_name_delta_value, self.array_multiply_udf(col(RBMCore.col_name_delta_value), lit(learning_rate))) \
            .withColumn(RBMCore.col_name_value, self.array_add_udf(col(RBMCore.col_name_old_value), col(RBMCore.col_name_delta_value)))\
            .select(RBMCore.col_name_item_index, RBMCore.col_name_hidden_node_index, RBMCore.col_name_value)


        # self._weight.persist(StorageLevel.MEMORY_AND_DISK)
        # self._visible_layer_bias.persist(StorageLevel.MEMORY_AND_DISK)
        # self._hidden_layer_bias.persist(StorageLevel.MEMORY_AND_DISK)
        # self._visible_layer_bias = self._visible_layer_bias.cache()
        # self._hidden_layer_bias = self._hidden_layer_bias.cache()
        # self._weight = self._weight.cache()
        self._visible_layer_bias = self._visible_layer_bias.cache()
        self._hidden_layer_bias = self._hidden_layer_bias.cache()
        self._weight = self._weight.cache()
        self._visible_layer_bias = self._visible_layer_bias.checkpoint(True)
        self._hidden_layer_bias = self._hidden_layer_bias.checkpoint(True)
        self._weight = self._weight.checkpoint(True)
        # self._weight.explain()
        # print(self._weight.rdd.toDebugString())
        # self._weight.count()
        # self._hidden_layer_bias.count()
        # self._visible_layer_bias.count()
        # self._weight.explain()
        # print(self._weight.rdd.toDebugString())
        # dw = dw.unpersist()
        # db_v = db_v.unpersist()
        # db_h = db_h.unpersist()
        # print(self._visible_layer_bias.count())
        # print(self._hidden_layer_bias.count())
        # print(self._weight.count())

        # self.checkPoint()
        # print("After update w,v_b,h_b count: {},{},{}".format(
        #     str(self._weight.count()), str(self._visible_layer_bias.count()), str(self._hidden_layer_bias.count())))
        # print("done update_parameter")
        return self._weight, self._visible_layer_bias, self._hidden_layer_bias


class RBM(HasUserCol, HasItemCol, HasValueCol, HasNumberOfHiddenNode, HasSoftMaxUnit,
          HasLearningRate, HasIterNum, Estimator):

    def __init__(self, spark, cd_list=[1, 3, 5],cd_iter=[0.5, 0.8,1] ):
        super(RBM, self).__init__()
        self.spark = spark
        self.random_numpy_udf = udf(random_numpy, ArrayType(DoubleType(), containsNull=False))
        self.weight = None
        self.bv = None
        self.bh = None
        self.item_mapping = None # 2 col (self.getItemCol(),col_name_item_index)
        self.cd_list = cd_list
        self.cd_iter = cd_iter

    def _load_json_data(self, path):
        return self.spark.read \
            .option("header", "true") \
            .option("treatEmptyValuesAsNulls", "true") \
            .option("inferSchema", "true") \
            .option("charset", "UTF-8") \
            .json(path)

    def load(self, weight_path, bv_path, bh_path, item_mapping_path):
        self.weight = self._load_json_data(weight_path)
        self.bv = self._load_json_data(bv_path)
        self.bh = self._load_json_data(bh_path)
        self.item_mapping = self._load_json_data(item_mapping_path)
        return self

    def create_model_from_file(self,  weight_path, bv_path, bh_path, item_mapping_path):
        weight = self._load_json_data(weight_path)
        bv = self._load_json_data(bv_path)
        bh = self._load_json_data(bh_path)
        item_mapping = self._load_json_data(item_mapping_path)
        return RBMCore(self.spark, weight, bv, bh, item_mapping)\
            .setUserCol(self.getUserCol())\
            .setItemCol(self.getItemCol())\
            .setSoftMaxUnit(self.getSoftMaxUnit())\
            .setValueCol(self.getValueCol())\
            .setNumberOfHiddenNode(self.getNumberOfHiddenNode())

    #dataset (UserCol, ItemCol, SoftMaxIndexCol, ValueCol)
    def _fit(self, dataset):
        # some column name
        col_name_item_index = "i"
        col_name_hidden_node_index = "j"
        col_name_value = "value"
        col_name_value_visible = "value_visible"
        col_name_value_hidden = "value_hidden"
        col_name_value_weight = "value_weight"
        col_name_mul = "weight_mul_value"
        col_name_probability = "probability"
        col_name_nominator = "nominator"
        col_name_denominator = "denominator"
        col_name_delta_value = "delta_value"
        col_name_old_value = "old_value"
        col_name_value_0 = "value_0"
        col_name_value_k = "value_k"
        col_name_activation = "activation"

        npartition = 1024
        self.spark.conf.set("spark.sql.shuffle.partitions", npartition)

        cd_list = self.cd_list
        cd_iter = self.getIterNum() * np.array(self.cd_iter)
        # index item col
        item_mapping = None
        weight = None
        bh = None
        bv = None

        self.spark.conf.set("spark.sql.broadcastTimeout", "36000")


        # get some count

        # get all disctinct user
        user_df = dataset.select(self.getUserCol()).distinct().cache()
        number_of_user = user_df.count()
        # init train parameter
        if self.bh is None or self.bv is None or self.weight is None or self.item_mapping is None: #fitting at random value
            print("#########Index item col#########")
            item_df = dataset.select(self.getItemCol()).distinct()
            item_indexer = StringIndexer().setInputCol(self.getItemCol()).setOutputCol(col_name_item_index)
            item_index_model = item_indexer.fit(item_df)
            item_mapping = item_index_model.transform(item_df).cache()

            number_of_item = item_df.count()
            print("Number of item: " + str(number_of_item))

            print("#########Init weight, bias#########")
            bh = self.spark.createDataFrame([[x] for x in range(self.getNumberOfHiddenNode())], [col_name_hidden_node_index]) \
                .withColumn(col_name_value, lit(0.0)) \
                .repartition(col_name_hidden_node_index) \
                .cache()
            # .withColumn(col_name_value, rand() - 0.5)\
            # .persist(StorageLevel.MEMORY_AND_DISK)  # bias of hidden node
            bh.printSchema()
            bh.show()
            # bias of visible layer
            # for each item index there are K number of softmax unit
            # (col_name_item_index, col_name_soft_max_unit_index, col_name_value)
            # bv = self.spark.createDataFrame([[x] for x in range(number_of_item)], [col_name_item_index])
            # bv.show()
            # print(self.getSoftMaxUnit())
            bv = self.spark.createDataFrame([[x] for x in range(number_of_item)], [col_name_item_index])\
                .withColumn(col_name_value, array([lit(0.0) for x in range(self.getSoftMaxUnit())]))\
                .repartition(npartition, col_name_item_index) \
                .cache()
            # .persist(StorageLevel.MEMORY_AND_DISK)
            bv.printSchema()
            bv.show()
            # bv = dataset.select(self.getItemCol()).withColumn(col_name_value, np.random.rand(len(self.getSoftMaxUnit()))) #bias of visible node (dataframe(item, value))
            # weight
            weight = self.spark.createDataFrame([[x] for x in range(number_of_item)], [col_name_item_index]) \
                .crossJoin(broadcast(self.spark.createDataFrame([[x] for x in range(self.getNumberOfHiddenNode())], [col_name_hidden_node_index]))) \
                .withColumn(col_name_value, self.random_numpy_udf(lit(0.0), lit(0.01), lit(self.getSoftMaxUnit())))\
                .repartition(npartition, [col_name_item_index, col_name_hidden_node_index]) \
                .cache()
                # .persist(StorageLevel.MEMORY_AND_DISK)
            weight.printSchema()
            weight.show()

            weight.coalesce(1).write.format('json').save('dist/model/0/weight')
            bv.coalesce(1).write.format('json').save('dist/model/0/bv')
            bh.coalesce(1).write.format('json').save('dist/model/0/bh')
            item_mapping.coalesce(1).write.format('json').save("dist/model/0/item_mapping")
        else: #start fitting at load weight
            print("Load weight from file")
            bh = self.bh.repartition(npartition,col_name_hidden_node_index)
            bv = self.bv.repartition(npartition, col_name_item_index)
            weight = self.weight.repartition(npartition, [col_name_item_index, col_name_hidden_node_index])
            item_mapping = self.item_mapping

        rbm_core = RBMCore(self.spark, weight, bv, bh, item_mapping)\
            .setUserCol(self.getUserCol())\
            .setItemCol(self.getItemCol())\
            .setSoftMaxUnit(self.getSoftMaxUnit())\
            .setValueCol(self.getValueCol())\
            .setNumberOfHiddenNode(self.getNumberOfHiddenNode())

        print("#########Train#########")
        data_checkpoint_location = rbm_core.hdfs_path + "data_rbm"
        self.spark.sparkContext.setCheckpointDir(data_checkpoint_location)
        bv.show()
        bh.show()
        weight.show()
        item_mapping.show()
        training_df = dataset.join(item_mapping, [self.getItemCol()]) \
            .withColumn(col_name_item_index, col(col_name_item_index).cast(LongType()))\
            .withColumnRenamed(self.getValueCol(), col_name_value) \
            .select(self.getUserCol(), col_name_item_index, col_name_value)
        training_df = self.spark.createDataFrame(training_df.rdd, training_df.schema).cache()
        training_df.count()
        # training_df.explain()
        # user_list = training_df.select(self.getUserCol()).distinct().rdd.flatMap(lambda x: x).collect()
        # user_mini_batch = [user_list[i:i + 2500] for i in range(0, len(user_list), 2500)]
        # user_df = [self.spark.createDataFrame([[user] for user in user_mini_batch[i]], [self.getUserCol()]) for i in range(len(user_mini_batch))]
        # for i in range(len(user_df)):
        #     user_df[i].printSchema()

        # user_data_df = [training_df.join(user_df[i], [self.getUserCol()]).cache().checkpoint(True) for i in range(len(user_df))]
        # user_data_df = [training_df.checkpoint(True).cache()]
        for iter_i in range(self.getIterNum()):
            current_checkpoint_dir = rbm_core.hdfs_path + str(iter_i)
            self.spark.sparkContext.setCheckpointDir(current_checkpoint_dir)
            cd = 1
            # choose gibb sampling step
            for index, cd_i in enumerate(cd_list):
                if iter_i < cd_iter[index]:
                    cd = cd_i
                    break
            # count = 0
            print("Begin " + str(iter_i) + " iteration, cd: " + str(cd))
            start = time.time()
            # for user_data_index, user_data in enumerate(user_data_df):
                # count += 1
                # print(weight.rdd.getNumPartitions())
                # print(bv.rdd.getNumPartitions())
                # print(bh.rdd.getNumPartitions())
            vk, ph0_df, phk_df = rbm_core.gibbs_sampling(training_df,
                                                    cd, False)  # number of cycle gibb sampling is 3 (placeholder)
                # dw_df, dvb_df, dhb_df = rbm_core.compute_gradient(user_data, vk, ph0_df, phk_df, len(user_mini_batch[user_data_index]))
            dw_df, dvb_df, dhb_df = rbm_core.compute_gradient(training_df, vk, ph0_df, phk_df, number_of_user)
            weight, bv, bh = rbm_core.update_parameter(dw_df, dvb_df, dhb_df, self.getLearningRate())
            vk = vk.unpersist()
            ph0_df = ph0_df.unpersist()
            phk_df = phk_df.unpersist()
            vk = None
            ph0_df = None
            phk_df = None
            dw_df = None
            dvb_df=None
            dhb_df=None
            #save weight after 10 epoch
            if iter_i % 10 == 0 and iter_i != 0:
                weight.coalesce(1).write.format('json').save('dist/model/' + str(iter_i) + '/weight')
                bv.coalesce(1).write.format('json').save('dist/model/' + str(iter_i) + '/bv')
                bh.coalesce(1).write.format('json').save('dist/model/' + str(iter_i) + '/bh')
                item_mapping.coalesce(1).write.format('json').save("dist/model/" + str(iter_i) + '/item_mapping')
            #     weight = None
            #     bv = None
            #     bh = None
            # else:
            #     weight = None
            #     bv = None
            #     bh = None
            self.spark.sparkContext._jvm.System.gc()

                # print(weight.rdd.getNumPartitions())
                # print(bv.rdd.getNumPartitions())
                # print(bh.rdd.getNumPartitions())

            #     if count == 3:
            #         count = 0
            #         rbm_core.checkPoint
            # rbm_core.checkPoint()
            #remove previous checkpoint
            if iter_i > 0:
                previous_iter = iter_i - 1
                previous_checkpoint_dir = rbm_core.hdfs_path + str(previous_iter)
                # print(["hdfs", "dfs", "-rm", "-R", previous_checkpoint_dir])
                subprocess.call(["/opt/hadoop-2.7.7/bin/hdfs", "dfs", "-rm", "-R", previous_checkpoint_dir])
            end = time.time()
            print("Execution iteration: %s took %s seconds" % (iter_i, end - start))

        print("begin execute and write")
        # weight.show()
        # bv.show()
        # bh.show()
        weight.coalesce(1).write.format('json').save('dist/model/final/weight')
        bv.coalesce(1).write.format('json').save('dist/model/final/bv')
        bh.coalesce(1).write.format('json').save('dist/model/final/bh')
        item_mapping.coalesce(1).write.format('json').save("dist/model/final/item_mapping")
        return rbm_core


def set_array_value(input_array, set_index):
    input_array[set_index] = 1
    return input_array


def get_expectation(input_array):
    total = 0
    for index, val in enumerate(input_array):
        total += index * val
    return total


def get_max_index(input_array):
    return np.argmax(input_array).tolist()


class ValueToBinarySoftMaxModel(HasItemCol, HasValueCol, HasSoftMaxUnit, HasOutputCol, Transformer):

    def __init__(self, spark):
        super(ValueToBinarySoftMaxModel, self).__init__()
        self.spark = spark
        self.set_array_value_udf = udf(set_array_value, ArrayType(LongType(), containsNull=False))

    def _transform(self, dataset):
        return dataset.withColumn(self.getOutputCol(), array([lit(0) for x in range(self.getSoftMaxUnit())]))\
                .withColumn(self.getOutputCol(), self.set_array_value_udf(col(self.getOutputCol()), col(self.getValueCol())))
            #se cho thang do doi name Active


class ProbabilitySoftMaxToExpectationModel(HasUserCol, HasItemCol, HasValueCol, HasOutputCol, Transformer):

    def __init__(self, spark):
        super(ProbabilitySoftMaxToExpectationModel, self).__init__()
        self.spark = spark
        self.get_expectation_udf = udf(get_expectation, DoubleType())

    def _transform(self, dataset):
        return dataset.withColumn(self.getOutputCol(), self.get_expectation_udf(col(self.getValueCol())))
        # return dataset.withColumn("Value", when(col(self.getValueCol()) == 1, col(self.getSoftMaxIndexCol())).otherwise(0))
        # se cho thang do doi name Value va can drop duplicate


class ProbabilitySoftMaxGetMaxIndexModel(HasUserCol, HasItemCol, HasValueCol, HasOutputCol, Transformer):

    def __init__(self, spark):
        super(ProbabilitySoftMaxGetMaxIndexModel, self).__init__()
        self.spark = spark
        self.get_max_index_udf = udf(get_max_index, LongType())

    def _transform(self, dataset):
        return dataset.withColumn(self.getOutputCol(), self.get_max_index_udf(col(self.getValueCol())))
        # return dataset.withColumn("Value", when(col(self.getValueCol()) == 1, col(self.getSoftMaxIndexCol())).otherwise(0))
        # se cho thang do doi name Value va can drop duplicate