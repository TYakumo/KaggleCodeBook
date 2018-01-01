import    numpy     as       np
import    time
import    DatasetVectorManager      as VectorManager
import    MetricScorer              as MetricScorer
import    copy
from      sklearn   import   metrics
from      sklearn   import   svm
from      sklearn   import   cross_validation
from      random    import   randrange, choice


class SelfDefinedCrossValidator:
    def __init__(self):
        pass

    def setCVData(self, classifier, mlDataset):
        self.setClassifier(classifier)
        self.setMLDataset(mlDataset)

    def setClassifier(self, classifier):
        self.classifier = classifier

    def setMLDataset(self, mlDataset):
        self.mlDataset = mlDataset

    def performCrossValidation(self, iter_times):
        print "cross validation initializing ..."
        n_samples = self.mlDataset.getDatasetSize()
        rcv = cross_validation.ShuffleSplit(n_samples, n_iter=iter_times,
        test_size=0.2)
        self.doCrossValidation(rcv=rcv)

    def doCrossValidation(self, rcv):
        cv_scores = []
        nowIter = 0
        scorer = MetricScorer.MetricScorer()

        for train_index, test_index in rcv:
            print 'Cross Validation ' + str(nowIter)
            startTime = time.time()

            cvTrainDataset = self.mlDataset.selectByIndexList(train_index)
            cvTestDataset  = self.mlDataset.selectByIndexList(test_index)

            cvTrainVector = VectorManager.DatasetVectorManager(cvTrainDataset)
            cvTrainVector.generateTrainVector()

            cvTestVector = VectorManager.DatasetVectorManager(cvTestDataset)
            cvTestVector.generateCVTestVector()

            self.classifier.fitData(trainData=cvTrainVector)
        #    CVanswer = np.array([int(x[11]) for x in cv_testDataset])
        #    probas = classifier.getProba(cv_test)[:,1]
        #    res = metrics.roc_auc_score(CVanswer, probas)
        #    cv_scores.append(res)
            stopTime = time.time()
            print 'Execution time : ' + str( (stopTime-startTime) )
            print cvTestVector.getTargetVector()
            print self.classifier.getResult(testData=cvTestVector)
            res = scorer.normalizedGiniScore(cvTestVector.getTargetVector(), self.classifier.getResult(testData=cvTestVector))
            print 'Result : ' + str(res)
            cv_scores.append(res)

            nowIter = nowIter+1

        print cv_scores
        print np.average(cv_scores)
        print np.std(cv_scores)
