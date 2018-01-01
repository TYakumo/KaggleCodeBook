class DatasetVectorManager:
    def __init__(self, mlDataset):
        self.mlDataset = mlDataset

    def generateTrainVector(self, generateTarget=True):
        print 'Generating Training ML vectors ...'
        rawDataset = self.mlDataset.getDataset("rawDataset")
        self.vectorValue = rawDataset.values[:,2:]

        if generateTarget == True:
            self.targetValue = rawDataset.values[:,1]

        print 'Total : ' + str(len(self.vectorValue)) + ' records'

    def generateCVTestVector(self):
        self.generateTrainVector(generateTarget=True)

    def getVectorSize(self):
        return mlDataset.size

    def generateTestVector(self):
        print 'Generating Testing ML vectors ...'
        rawDataset = self.mlDataset.getDataset("rawDataset")
        self.vectorValue = rawDataset.values[:,1:]
        for valueList in self.vectorValue:
            for value in valueList:
                value = float(value)
        print 'Total  : ' + str(len(self.vectorValue)) + ' records'

    def getVector(self):
        return self.vectorValue

    def getTargetVector(self):
        return self.targetValue

    def printDataForDebug(self):
        print self.vectorValue
