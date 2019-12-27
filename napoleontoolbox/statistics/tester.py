from statsmodels.tsa.stattools import adfuller

def testStationarityDickeyFuller(X, confidenceLevel = '1%', displayResult = False):
    result = adfuller(X)
    if displayResult:
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        print('Critical Values:')
        for key, value in result[4].items():
            print('\t%s: %.3f' % (key, value))

    if confidenceLevel in result[4].keys():
        threshold = result[4][confidenceLevel]
        if result[0]<=threshold:
            return True
        else :
            return False
    else :
        raise Exception('Threshold '+confidenceLevel+' not available')



