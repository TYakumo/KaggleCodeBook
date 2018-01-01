from     sklearn.tree     import DecisionTreeRegressor
from     sklearn.ensemble import RandomForestRegressor
from     sklearn.ensemble import GradientBoostingRegressor
from     sklearn.ensemble import AdaBoostRegressor
#Base Regressor
class RegressorBase:
    def __init__(self, learningParam):
        self.regressor = self.makeRegressor(learningParam=learningParam)

    def remakeRegressor(self, learningParam):
        self.regressor = makeRegressor(learningParam=learningParam)

    def makeRegressor(self, learningParam):
        currentRegressor = RandomForestRegressor(
                            max_depth=learningParam.getParam('max_depth'),
                            min_samples_split=learningParam.getParam('min_samples_split'),
                            min_samples_leaf=learningParam.getParam('min_samples_leaf'))
        return currentRegressor

    def fitData(self, trainData):
        print 'Start fitting data ...'
        self.regressor.fit(trainData.getVector(), trainData.getTargetVector())

    def getResult(self, testData):
        print 'Generating label results ...'
        return self.regressor.predict(testData.getVector())

class RegressorGradientBoosting(RegressorBase):
    def __init__(self, learningParam):
        self.regressor = self.makeRegressor(learningParam=learningParam)

    def remakeRegressor(self, learningParam):
        self.regressor = makeClassifier(learningParam=learningParam)

    def makeRegressor(self, learningParam):
        currentRegressor         = GradientBoostingRegressor(
            n_estimators         = learningParam.getParam('n_estimators'),
            learning_rate        = learningParam.getParam('learning_rate'),
            max_depth            = learningParam.getParam('max_depth'),
            subsample            = learningParam.getParam('subsample'),
            min_samples_split    = learningParam.getParam('min_samples_split'),
            min_samples_leaf     = learningParam.getParam('min_samples_leaf')
            #n_jobs = -1
            )
        return currentRegressor

class RegressorAdaboost(RegressorBase):
    def __init__(self, adaLearningParam):
        self.regressor = self.makeRegressor(adaLearningParam=adaLearningParam)

    def remakeRegressor(self, adaLearningParam):
        self.regressor = makeClassifier(adaLearningParam=adaLearningParam)

    def makeRegressor(self, adaLearningParam):
        currentRegressor = AdaBoostRegressor(
            base_estimator=RandomForestRegressor(
                n_estimators=adaLearningParam.getBaseParam('n_estimators'),
                max_depth=adaLearningParam.getBaseParam('max_depth'),
                min_samples_split=adaLearningParam.getBaseParam('min_samples_split'),
                min_samples_leaf=adaLearningParam.getBaseParam('min_samples_leaf'),
                n_jobs = -1
                ),
            n_estimators = adaLearningParam.getParam('n_estimators'),
            learning_rate = adaLearningParam.getParam('learning_rate'),
            )
        return currentRegressor

