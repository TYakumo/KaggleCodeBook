from       sklearn                   import preprocessing
import     sys
import     pandas                    as pd
import     MLDataset                 as MLData
import     ClassifierClasses         as CLFS
import     DatasetVectorManager      as VectorManager
import     ResultOutputer            as ResultOutputer
import     SelfDefinedCrossValidator as CValidator
import     DataPreprocessor          as DataPreprocessor

def loadData(mlDataset, mlTestDataset):
    print 'Loading data'
    rawDataset = pd.read_csv('./Data/train_ok.csv')
    mlDataset.addMainDataset(dataset=rawDataset)

    rawTestDataset = pd.read_csv('./Data/test_ok.csv')
    mlTestDataset.addMainDataset(dataset=rawTestDataset)

def doCrossValidation(mlDataset, classifier):
    cv = CValidator.SelfDefinedCrossValidator()
    cv.setCVData(mlDataset=mlDataset, classifier=classifier)
    cv.performCrossValidation(iter_times=5)

def generateResult(dataVector, testDataVector, classifier, finalOutput):
    classifier.fitData(trainData=dataVector)
    finalOutput.outputResult( clfResult=classifier.getResult(testData=testDataVector) )

def parseArgv(argvDict):
    for argValue in sys.argv:
        argvDict[ argValue ] = True

def main():
    trainEncoderList = []
    testEncoderList = []
    argvDict = {}
    parseArgv(argvDict=argvDict)

    finalOutput = ResultOutputer.ResultOutputer()

    mlDataset = MLData.MLDataset()
    mlTestDataset = MLData.MLDataset()
    loadData(mlDataset=mlDataset, mlTestDataset=mlTestDataset)

    dataVector = VectorManager.DatasetVectorManager(mlDataset=mlDataset)
    dataVector.generateTrainVector()

    testDataVector = VectorManager.DatasetVectorManager(mlDataset=mlTestDataset)
    testDataVector.generateTestVector()

    classifier = CLFS.ClassifierAdaboost()

    if "benchmark" in argvDict:
        doCrossValidation(mlDataset=mlDataset, classifier=classifier)

    if "generateResult" in argvDict:
        generateResult(dataVector=dataVector, testDataVector=testDataVector, classifier=classifier, finalOutput=finalOutput)

if __name__=="__main__":
    main()


