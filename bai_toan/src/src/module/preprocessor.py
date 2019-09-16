from pyspark.sql.functions import col, count, lit, rank
from .custom_params import HasItemCol, HasOutputCol, HasValueCol, HasUserCol
from pyspark.ml.pipeline import Estimator, Transformer
from pyspark.sql.window import Window


class FullCourseMappingEstimator(HasItemCol, HasOutputCol, Estimator):

    def __init__(self):
        super(FullCourseMappingEstimator, self).__init__()

    #dataset (UserCol, ItemCol, SoftMaxIndexCol, ValueCol)
    def _fit(self, mapping_df):
        self.mapping_df = mapping_df.select(self.getItemCol(), self.getOutputCol()).distinct()
        return FullCourseMappingTransformer(self.mapping_df)\
            .setItemCol(self.getItemCol())\
            .setOutputCol(self.getOutputCol())


class FullCourseMappingTransformer(HasItemCol, HasOutputCol, Transformer):

    def __init__(self, mapping_df):
        super(FullCourseMappingTransformer, self).__init__()
        self.mapping_df = mapping_df

    def _transform(self, dataset):
        item_df = dataset.select(self.getItemCol()).distinct()
        missing_item_df = item_df.subtract(self.mapping_df.select(self.getItemCol()).distinct())\
            .withColumn(self.getOutputCol(), col(self.getItemCol()))
        full_mapping_df = missing_item_df\
            .unionAll(self.mapping_df)
        return dataset.join(full_mapping_df, [self.getItemCol()])


class FilterCountTransformer(HasItemCol, Transformer):

    def __init__(self, limit=5):
        super(FilterCountTransformer, self).__init__()
        self.limit = limit

    def _transform(self, dataset):
        item_count_df = dataset.groupBy(self.getItemCol())\
            .agg(count(lit(1)).alias("count"))\
            .filter(col("count") > self.limit)\
            .drop("count")
        return dataset.join(item_count_df, [self.getItemCol()])


class FilterDuplicateUserItemGetMaxTransformer(HasUserCol, HasItemCol, HasValueCol, Transformer):

    def __init__(self):
        super(FilterDuplicateUserItemGetMaxTransformer, self).__init__()

    def _transform(self, dataset):
        window = Window.partitionBy([col(self.getItemCol()), self.getUserCol()]) \
            .orderBy(col(self.getValueCol()).desc())

        dataset = dataset.select("*", rank().over(window).alias("rank")) \
            .filter(col("rank") == 1).distinct()

        dataset = dataset.distinct()
        return dataset


class FilterDuplicateMappingTransformer(HasItemCol, HasValueCol, Transformer):

    def __init__(self):
        super(FilterDuplicateMappingTransformer, self).__init__()

    def _transform(self, dataset):
        fail = False
        distinct_mapping = dataset.select(self.getItemCol(), self.getValueCol()).distinct()

        left_df = distinct_mapping.select(self.getItemCol())
        right_df = distinct_mapping.select(self.getValueCol())

        check_df = left_df.join(right_df.withColumnRenamed(self.getValueCol(), self.getItemCol()), [self.getItemCol()])
        check_count = check_df.count()
        print(check_count)
        if check_count > 0:
            print("Alert mapping conflict")
            check_df.show()
            fail = True

        duplicate_left_df = left_df.groupBy(self.getItemCol()).agg(count(lit(1)).alias("duplicate_count"))\
            .filter(col("duplicate_count") > 1)
        duplicate_left_count = duplicate_left_df.count()
        if duplicate_left_count > 0:
            print("Alert multiple value left")
            duplicate_left_df.show()
            fail = True

        if fail is True:
            return None
        else:
            return distinct_mapping

#
# class FilterNonEduCourseTransformer(HasItemCol, Transformer):
#
#     def __init__(self, item_filter_out_df):
#         super(FilterNonEduCourseTransformer, self).__init__()
#         self.item_filter_out_df = item_filter_out_df
#
#     def _transform(self, dataset):
#         item_df = dataset.select(self.getItemCol()).distinct()
#         valid_item_df = item_df.subtract(self.item_filter_out_df).distinct()
#         return_df = dataset.join()
#         valid_item_df = dataset.subtract(self.mapping_df.select(self.getItemCol()).distinct())\
#             .withColumn(self.getOutputCol(), col(self.getItemCol()))
#
