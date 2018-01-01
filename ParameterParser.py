import     LearningParameter         as LParam


class ParameterUtil:
    def __init__(self):
        pass

    @staticmethod
    def isInIntegerList(val):
        return val in {'max_depth':0, 'min_samples_split':0, 'min_samples_leaf':0, 'n_estimators':0}

    @staticmethod
    def isFloatList(val):
        return val in {'subsample':0, 'learning_rate':0}

class ParameterParser:
    def __init__(self):
        pass

    @staticmethod
    def updateParamByArgv(argvDict, learningParam):
        for x in argvDict:
            parseList = x.split("=")
            if ParameterUtil.isInIntegerList(parseList[0]):
                learningParam.setParam(parseList[0], int(parseList[1]))

            if ParameterUtil.isFloatList(parseList[0]):
                learningParam.setParam(parseList[0], float(parseList[1]))

    @staticmethod
    def updateAdaParamByArgv(argvDict, learningParam):
        pass
