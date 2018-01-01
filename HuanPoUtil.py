import    numpy     as       np

class HuanPoUtil:
    def __init__(self):
        pass

    @staticmethod
    def convertFloatListToIntListBySortedValue(arrayList):
        dataType = [('index', int), ('targetValue', float), ('rankValue', int)]

        lenArray = len(arrayList)
        newArrayList = [ (idx, value, 0) for idx, value in zip(xrange(lenArray), arrayList) ]
        npArray = np.array(newArrayList, dtype=dataType)
        npArray = np.sort(npArray, order='targetValue') #It is not reference !!
        preValue = None
        accumulation = 0
        groupCount = 0

        for x in npArray:
            if preValue == None or x['targetValue'] != preValue:
                preValue = x['targetValue']
                accumulation = accumulation + groupCount
                groupCount = 1
            else:
                groupCount = groupCount+1

            x['rankValue'] = accumulation

        npArray = np.sort(npArray, order='index') #It is not reference !!
        retList = [ rankValue for (idx, tv, rankValue) in npArray]
        return retList