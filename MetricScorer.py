import    numpy      as       np
from      HuanPoUtil import   HuanPoUtil
from      sklearn    import   metrics

class MetricScorer:
    def __init__(self):
        self.metricsScorer = self.normalizedGiniScore #FIXME, not working, don't know why

    def setMetrics(self, metricsScorer):
        self.metricsScorer = metricsScorer

    def giniScore(self, answerList, resultList):
        df = zip(answerList, resultList, range(len(answerList)))
        df = sorted(df, key=lambda x: (x[1],-x[2]), reverse=True)
        rand = [float(i+1)/float(len(df)) for i in range(len(df))]
        totalPos = float(sum([x[0] for x in df]))
        cumPosFound = [df[0][0]]
        for i in range(1,len(df)):
            cumPosFound.append(cumPosFound[len(cumPosFound)-1] + df[i][0])
        Lorentz = [float(x)/totalPos for x in cumPosFound]
        Gini = [Lorentz[i]-rand[i] for i in range(len(df))]
        return sum(Gini)

    def normalizedGiniScore(self, answerList, resultList):
        result = self.giniScore(answerList, resultList)/self.giniScore(answerList, answerList)
        return result

    def getScore(self, resultList, answerList):
        return self.metricsScorer
