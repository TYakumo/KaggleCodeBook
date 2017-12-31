class DatasetVectorManager:
    def __init__(self, mlDataset):
        self.mlDataset = mlDataset

    def generateTrainVector(self, generateTarget=True):
        print 'Generating Training ML vectors ...'
        rawDataset = self.mlDataset.getDataset("rawDataset")
        self.vectorValue = rawDataset.values[:,1:]

        if generateTarget == True:
            self.targetValue = rawDataset.values[:,0]

        print 'Total : ' + str(len(self.vectorValue)) + ' records'

    def generateCVTestVector(self):
        self.generateTrainVector(generateTarget=True)

    def getVectorSize(self):
        return mlDataset.size

    def generateTestVector(self):
        print 'Generating Testing ML vectors ...'
        rawDataset = self.mlDataset.getDataset("rawDataset")
        self.vectorValue = rawDataset.values[:,:]
        print 'Total  : ' + str(len(self.vectorValue)) + ' records'

    def getVector(self):
        return self.vectorValue

    def getTargetVector(self):
        return self.targetValue
