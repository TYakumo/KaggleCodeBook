import    random

class EasyEnsembleLearner:
	def __init__(self):
		self.predictors = []
        self.predictorSize = 0

	def setLearner(self, classifier):
		self.classifier = classifier

	def setData(mlDataset, mlTestDataset):
		self.mlDataset = mlDataset
		self.mlTestDataset = mlTestDataset

    def easyTrain(self):
        easyEnsembleTrainData = self.mlDataset
        self.ensembleClf = []
        minorSize = 5566

        while easyEnsembleTrainData.getDatasetSize() >= minorSize:
            totalList = range(self.easyEnsembleTrainData.getDatasetSize())
            random.shuffle(totalList)
            choiceList = totalList[0:minorSize]
            remainList = totalList[minorSize:]


            EETrainMLDataset = easyEnsembleTrainData.selectByIndexList(choiceList)
            EETrainDataVector = VectorManager.DatasetVectorManager(EETrainMLDataset)
            EETrainDataVector.generateTrainVector()

            EETestMLDataset  = easyEnsembleTrainData.selectByIndexList(remainList)
            EETestDataVector = VectorManager.DatasetVectorManager(EETestMLDataset)
            EETestDataVector.generateCVTestVector()

            wrongList = []
            resultList = self.classifier.getResult(testData=EETestDataVector)

            lenRL = len(resultList)
            for idx,clfR in zip(range(lenRL), resultList):
                if clfR != EETestDataVector.getTargetVector()[idx]:
                    wrongList.append(idx)

            easyEnsembleTrainData = easyEnsembleTrainData.selectByIndexList(wrongList)
            self.predictors.append( copy.deepcopy(self.classifier) )
            self.predictorSize = self.predictorSize+1

    def ensembleThreshold(self, nowValue, thresholdValue)
        return nowValue >= thresholdValue

	def getResult(self, testData):        
        print 'Generating label results ...'
        resultList = [0] * testData.getVectorSize()
        for clf in self.predictors:
            curList = clf.getResult(testData=testData)

        predictorSize = len(self.predictors)
        resultList = [x/float(predictorSize) for x in resultList]
        return resultList

	def getProbaResult(self, testData):
        print 'Generating probability results ...'  
        resultList = [0] * testData.getVectorSize()    
        for clf in self.predictors:
            curList = clf.getProbaResult(testData=testData)

        predictorSize = len(self.predictors)        
        halfPredictorSize = (predictorSize+1)/2
        resultList = [self.ensembleThreshold(x, halfPredictorSize) for x in resultList]
        return resultList