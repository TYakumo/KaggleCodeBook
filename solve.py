from       sklearn                   import preprocessing
import     sys
import     pandas                    as pd
import     MLDataset                 as MLData
import     ClassifierClasses         as CLFS
import     RegressorClasses          as RCLS
import     LearningParameter         as LParam
import     DatasetVectorManager      as VectorManager
import     ResultOutputer            as ResultOutputer
import     SelfDefinedCrossValidator as CValidator
import     DataPreprocessor          as DataPreprocessor
import     ParameterParser           as ParamParser

def loadData(mlDataset, mlTestDataset):
    print 'Loading data'
    rawDataset = pd.read_csv('./Data/train_ok.csv')
    mlDataset.addMainDataset(dataset=rawDataset)

    rawTestDataset = pd.read_csv('./Data/test_ok.csv')
    mlTestDataset.addMainDataset(dataset=rawTestDataset)

def doCrossValidation(mlDataset, classifier):
    cv = CValidator.SelfDefinedCrossValidator()
    cv.setCVData(mlDataset=mlDataset, classifier=classifier)
    cv.performCrossValidation(iter_times=20)

def generateResult(dataVector, testDataVector, classifier, finalOutput, indexList=None):

    classifier.fitData(trainData=dataVector)

    if indexList==None:
        finalOutput.outputResult( clfResult=classifier.getResult(testData=testDataVector) )
    else:
        finalOutput.outputResultWithCustomIndex( indexList=indexList, clfResult=classifier.getResult(testData=testDataVector) )

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

    indexList = mlTestDataset.getKeyList(dataKey='Id')
    #print indexList #For debug
    dataVector.printDataForDebug()
    testDataVector.printDataForDebug()

    #Parameter Initialization
    randomForestRegressorParameter = LParam.LearningParameter()
    randomForestRegressorParameter.initForRandomForestRegressor()
    ParamParser.ParameterParser.updateParamByArgv(argvDict, randomForestRegressorParameter)
    randomForestRegressorParameter.printParam()

    gradientBoostingRegressorParameter = LParam.LearningParameter()
    gradientBoostingRegressorParameter.initForGradientBoostingRegressor()
    ParamParser.ParameterParser.updateParamByArgv(argvDict, gradientBoostingRegressorParameter)
    gradientBoostingRegressorParameter.printParam()

    adaboostRegressorParameter = LParam.LearningParameterForAdaboost()
    adaboostRegressorParameter.initForRandomForestRegressorCombo()
    adaboostRegressorParameter.printParam()

    #Classifier or Regressor generation
    adaBoostRegressor = RCLS.RegressorAdaboost(adaLearningParam=adaboostRegressorParameter)

    gradientBoostingRegressor = RCLS.RegressorGradientBoosting(learningParam=gradientBoostingRegressorParameter)

    randomForestRegressor = RCLS.RegressorBase(learningParam=randomForestRegressorParameter)


    #Regressor selection
    regressor = randomForestRegressor

    if "gradientBoost" in argvDict:
        regressor = gradientBoostingRegressor

    if "adaBoost" in argvDict:
        regressor = adaBoostRegressor

    #Benchmark & Result
    if "benchmark" in argvDict:
        doCrossValidation(mlDataset=mlDataset, classifier=regressor)

    if "generateResult" in argvDict:
        generateResult(dataVector=dataVector, testDataVector=testDataVector, classifier=regressor, finalOutput=finalOutput, indexList=indexList)

if __name__=="__main__":
    main()


