import    pandas as pd
import    copy

class MLDataset:

    def __init__(self):
        self.datasetDict = {}
        self.datasetSize = -1

    def getDatasetSize(self):
        return self.datasetSize

    def addMainDataset(self, dataset):
        self.datasetSize = len(dataset)
        self.addDataset(dataKey='rawDataset', dataset=dataset)

    def selectByIndexList(self, selectList):
        currentDataset = copy.deepcopy(self)
        rawDataset = currentDataset.getMainDataset()
        currentDataset.datasetDict[ 'rawDataset' ] = rawDataset [ rawDataset.index.isin(selectList) ]
        currentDataset.datasetSize = currentDataset.datasetDict[ 'rawDataset' ].size
        return currentDataset

    def addDataset(self, dataKey, dataset):
        self.datasetDict[ dataKey ] = dataset
        if dataKey == 'rawDataset':
            self.datasetSize = len(dataset)

    def getMainDataset(self):
        return self.getDataset(dataKey='rawDataset')

    def getDataset(self, dataKey):
        return self.datasetDict[ dataKey ]

    def preProcessData(self, mlPreprocessor, dataColumnKey):
        for columnKey in dataColumnKey:
            for mlDataIdx in xrange(self.datasetSize):
                self.getMainDataset().loc[mlDataIdx, columnKey ] = mlPreprocessor.getEncoder(columnKey).transform( self.getMainDataset().loc[mlDataIdx, columnKey ] )

    def printDataForDebug(self):
        print self.getMainDataset()

    def saveToCsv(self, filePath):
        print 'Saving ' + str(filePath)
        self.getMainDataset().to_csv(path_or_buf=filePath, index=False)
        print 'Data Saved'
