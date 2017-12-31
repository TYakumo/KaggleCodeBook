from     sklearn.ensemble import AdaBoostClassifier
from     sklearn.ensemble import RandomForestClassifier
from     sklearn.ensemble import ExtraTreesClassifier
from     sklearn.ensemble import GradientBoostingClassifier

#Base classifier using a decision tree
class ClassifierBase:
    def __init__(self):
        self.classifier = self.makeClassifier()

    def remakeClassifier(self, max_depth=10):
        self.classifier = makeClassifier(max_depth=max_depth)

    def makeClassifier(self, max_depth=10):
        return currentClassifier

    def fitData(self, trainData):
        print 'Start fitting data ...'
        self.classifier.fit(trainData.getVector(), trainData.getTargetVector())

    def getProba(self, testData):        
        print 'Generating probability results ...'
        return self.classifier.predict_proba(testData.getVector())

    def getResult(self, testData):        
        print 'Generating label results ...'
        return self.classifier.predict(testData.getVector())

class ClassifierAdaboost(ClassifierBase):    
    def remakeClassifier(self, max_depth=10, adaboostLearner=100):
        self.classifier = makeClassifier(max_depth=max_depth, adaboostLearner=adaboostLearner)

    def makeClassifier(self, max_depth=10, adaboostLearner=100):
        currentClassifier = AdaBoostClassifier(
            n_estimators = adaboostLearner,
            learning_rate = 0.60,
            base_estimator = ExtraTreesClassifier(
                n_estimators = 200,
                max_depth = max_depth,
                min_samples_leaf = 200,
                min_samples_split = 50,
                n_jobs = -1)
            )
        return currentClassifier