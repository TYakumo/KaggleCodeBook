from      sklearn     import preprocessing
from      numpy       import genfromtxt, savetxt
import    MLDataset   as MLData
import    copy

class DataPreprocessor:

    def __init__(self):
        self.mlDatasetList = []
        self.dataColumnKeyList = []
        self.encoders      = {}

    def addDataset(self, mlDataset, dataColumnKey):
        self.mlDatasetList.append(mlDataset)
        self.dataColumnKeyList.append(dataColumnKey)

    def fitData(self):
        self.mlDataListLen = len(self.mlDatasetList)
        self.numOfEncoder  = len(self.dataColumnKeyList[0])

        dataSize = len(self.mlDatasetList)
        for datasetKeyIdx in xrange(self.numOfEncoder):
            for idx in xrange(dataSize):
                datasetKey = str(self.dataColumnKeyList[idx][datasetKeyIdx])

                trainList = []
                for mlDataset in self.mlDatasetList:
                    trainList = trainList + list(mlDataset.getMainDataset()[datasetKey].values)

            encoder = preprocessing.LabelEncoder()
            encoder.fit(trainList)
            self.encoders[datasetKey] = copy.deepcopy(encoder)

    def getEncoder(self, encoderKey):
        return self.encoders[encoderKey]
