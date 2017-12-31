from      numpy import genfromtxt, savetxt
import    numpy as np

class ResultOutputer:
    def __init__(self):
        pass

    def outputResult(self, clfResult):     
        finalResult = [['ImageId','Label']]

        for idx,resultValue in zip( range(len(clfResult)) , clfResult):
            finalResult.append([str(idx+1),str(float(resultValue))])
        savetxt('./Data/submission.csv', finalResult, fmt='%s,%s')
