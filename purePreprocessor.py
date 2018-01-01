from       sklearn                   import preprocessing
import     sys
import     pandas                    as pd
import     MLDataset                 as MLData
import     ClassifierClasses         as CLFS
import     DatasetVectorManager      as VectorManager
import     DataPreprocessor          as DataPreprocessor

def loadData(mlDataset, mlTestDataset):
    print 'Loading data'
    rawDataset = pd.read_csv('./Data/train.csv')
    mlDataset.addMainDataset(dataset=rawDataset)

    rawTestDataset = pd.read_csv('./Data/test.csv')
    mlTestDataset.addMainDataset(dataset=rawTestDataset)

def main():
    trainEncoderList = []
    testEncoderList = []

    mlDataset = MLData.MLDataset()
    mlTestDataset = MLData.MLDataset()
    loadData(mlDataset=mlDataset, mlTestDataset=mlTestDataset)

    dataColumnKey = ['T1_V4', 'T1_V5', 'T1_V6', 'T1_V7', 'T1_V8', 'T1_V9', 'T1_V11', 'T1_V12', 'T1_V15', 'T1_V16', 'T1_V17',
    'T2_V3', 'T2_V5', 'T2_V11', 'T2_V12', 'T2_V13']
    dataPreprocessor = DataPreprocessor.DataPreprocessor()
    dataPreprocessor.addDataset(mlDataset=mlDataset, dataColumnKey=dataColumnKey)
    dataPreprocessor.addDataset(mlDataset=mlTestDataset, dataColumnKey=dataColumnKey)
    dataPreprocessor.fitData()

    mlDataset.preProcessData(mlPreprocessor=dataPreprocessor, dataColumnKey=dataColumnKey)
    mlTestDataset.preProcessData(mlPreprocessor=dataPreprocessor, dataColumnKey=dataColumnKey)

    mlDataset.printDataForDebug()
    mlTestDataset.printDataForDebug()
    mlDataset.saveToCsv(filePath='./Data/train_ok.csv')
    mlTestDataset.saveToCsv(filePath='./Data/test_ok.csv')

if __name__=="__main__":
    main()


