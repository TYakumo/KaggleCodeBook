from      numpy import genfromtxt, savetxt
import    numpy as np

class ResultOutputer:
    def __init__(self):
        pass

    def outputResult(self, clfResult):
        finalResult = [['Id','Hazard']]

        for idx,resultValue in zip( range(len(clfResult)) , clfResult):
            finalResult.append([str(idx+1),str(float(resultValue))])
        savetxt('./Data/submission.csv', finalResult, fmt='%s,%s')

    def outputResultWithCustomIndex(self, indexList, clfResult):
        finalResult = [['Id','Hazard']]

        for idx,resultValue in zip( indexList, clfResult):
            finalResult.append([str(idx),str(max(float(resultValue), 0.0))])
        savetxt('./Data/submission.csv', finalResult, fmt='%s,%s')
