
class LearningParameter:
    def __init__(self):
        self.parameterDict = {}

    def initForRandomForestRegressor(self):
        self.parameterDict['learnerType']               = 'randomForestRegressor'
        self.parameterDict['max_depth']                 = 10
        self.parameterDict['min_samples_split']         = 50
        self.parameterDict['min_samples_leaf']          = 50
        self.parameterDict['n_estimators']              = 1500

    def initForRandomForestClassifier(self):
        self.parameterDict['learnerType']               = 'randomForestClassifier'
        self.parameterDict['max_depth']                 = 10
        self.parameterDict['min_samples_split']         = 50
        self.parameterDict['min_samples_leaf']          = 50
        self.parameterDict['n_estimators']              = 1500

    def initForGradientBoostingRegressor(self):
        self.parameterDict['learnerType']               = 'gradientBoostingRegressor'
        self.parameterDict['learning_rate']             = 0.1
        self.parameterDict['max_depth']                 = 12
        self.parameterDict['n_estimators']              = 1500
        self.parameterDict['subsample']                 = 1.0
        self.parameterDict['min_samples_split']         = 50
        self.parameterDict['min_samples_leaf']          = 50

    def initForGradientBoostingClassifier(self):

        self.parameterDict['learnerType']               = 'gradientBoostingClassifier'
        self.parameterDict['learning_rate']             = 0.1
        self.parameterDict['max_depth']                 = 12
        self.parameterDict['n_estimators']              = 1500
        self.parameterDict['subsample']                 = 1.0
        self.parameterDict['min_samples_split']         = 50
        self.parameterDict['min_samples_leaf']          = 50

    def getParam(self, parameterKey):
        return self.parameterDict[parameterKey]

    def setParam(self, parameterKey, parameterValue):
        self.parameterDict[parameterKey] = parameterValue

    def printParam(self): #for debug
        print self.parameterDict

class LearningParameterForAdaboost(LearningParameter):

    def __init__(self):
        self.parameterDict = {}
        self.baseParam = LearningParameter() #LearningParameter for base estimator
        self.parameterDict['learnerType']               = 'adaboost'

    def initForRandomForestRegressorCombo(self):
        self.baseParam.initForRandomForestRegressor()
        self.parameterDict['n_estimators']              = 50
        self.parameterDict['learning_rate']             = 1.0

    def initForGradientBoostingRegressorCombo(self):
        self.baseParam.initForGradientBoostingRegressor()
        self.parameterDict['n_estimators']              = 50
        self.parameterDict['learning_rate']             = 1.0

    def getBaseParam(self, parameterKey):
        self.baseParam.getParam(parameterKey)

    def setBaseParam(self, parameterKey, parameterValue):
        self.baseParam.setParam(parameterKey, parameterValue)
