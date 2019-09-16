from pyspark.ml.param.shared import Params
from pyspark.ml.param.shared import Param
from pyspark.ml.param.shared import TypeConverters


class HasSoftMaxUnit(Params):
    softMaxUnit = Param(Params._dummy(), "softMaxUnit", "softMaxUnit",
                        typeConverter=TypeConverters.toInt)

    def __init__(self):
        super(HasSoftMaxUnit, self).__init__()

    def setSoftMaxUnit(self, value):
        return self._set(softMaxUnit=value)

    def getSoftMaxUnit(self):
        return self.getOrDefault(self.softMaxUnit)


class HasRank(Params):
    rank = Param(Params._dummy(), "rank", "rank",
                        typeConverter=TypeConverters.toInt)

    def __init__(self):
        super(HasRank, self).__init__()

    def setRank(self, value):
        return self._set(rank=value)

    def getRank(self):
        return self.getOrDefault(self.rank)


class HasUserCol(Params):
    userCol = Param(Params._dummy(), "userCol", "userCol",
                    typeConverter=TypeConverters.toString)

    def __init__(self):
        super(HasUserCol, self).__init__()

    def setUserCol(self, value):
        return self._set(userCol=value)

    def getUserCol(self):
        return self.getOrDefault(self.userCol)


class HasItemCol(Params):
    itemCol = Param(Params._dummy(), "itemCol", "itemCol",
                    typeConverter=TypeConverters.toString)

    def __init__(self):
        super(HasItemCol, self).__init__()

    def setItemCol(self, value):
        return self._set(itemCol=value)

    def getItemCol(self):
        return self.getOrDefault(self.itemCol)


class HasSoftMaxIndexCol(Params):
    softMaxIndexCol = Param(Params._dummy(), "softMaxIndexCol", "softMaxIndexCol",
                            typeConverter=TypeConverters.toString)

    def __init__(self):
        super(HasSoftMaxIndexCol, self).__init__()

    def setSoftMaxIndexCol(self, value):
        return self._set(softMaxIndexCol=value)

    def getSoftMaxIndexCol(self):
        return self.getOrDefault(self.softMaxIndexCol)


class HasValueCol(Params):
    valueCol = Param(Params._dummy(), "valueCol", "valueCol",
                     typeConverter=TypeConverters.toString)

    def __init__(self):
        super(HasValueCol, self).__init__()

    def setValueCol(self, value):
        return self._set(valueCol=value)

    def getValueCol(self):
        return self.getOrDefault(self.valueCol)


class HasNumberOfHiddenNode(Params):
    numberOfHiddenNode = Param(Params._dummy(), "numberOfHiddenNode", "numberOfHiddenNode",
                               typeConverter=TypeConverters.toInt)

    def __init__(self):
        super(HasNumberOfHiddenNode, self).__init__()

    def setNumberOfHiddenNode(self, value):
        return self._set(numberOfHiddenNode=value)

    def getNumberOfHiddenNode(self):
        return self.getOrDefault(self.numberOfHiddenNode)


class HasLearningRate(Params):
    learningRate = Param(Params._dummy(), "learningRate", "learningRate",
                         typeConverter=TypeConverters.toFloat)

    def __init__(self):
        super(HasLearningRate, self).__init__()

    def setLearningRate(self, value):
        return self._set(learningRate=value)

    def getLearningRate(self):
        return self.getOrDefault(self.learningRate)


class HasPredictCol(Params):
    predictCol = Param(Params._dummy(), "predictCol", "predictCol",
                       typeConverter=TypeConverters.toString)

    def __init__(self):
        super(HasPredictCol, self).__init__()

    def setPredictCol(self, value):
        return self._set(predictCol=value)

    def getPredictCol(self):
        return self.getOrDefault(self.predictCol)


class HasOutputCol(Params):
    outputCol = Param(Params._dummy(), "outputCol", "outputCol",
                      typeConverter=TypeConverters.toString)

    def __init__(self):
        super(HasOutputCol, self).__init__()

    def setOutputCol(self, value):
        return self._set(outputCol=value)

    def getOutputCol(self):
        return self.getOrDefault(self.outputCol)


class HasIterNum(Params):
    iterNum = Param(Params._dummy(), "iterNum", "iterNum",
                      typeConverter=TypeConverters.toInt)

    def __init__(self):
        super(HasIterNum, self).__init__()

    def setIterNum(self, value):
        return self._set(iterNum=value)

    def getIterNum(self):
        return self.getOrDefault(self.iterNum)
