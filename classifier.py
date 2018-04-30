from scipy.spatial import distance
from sklearn.metrics import accuracy_score
from scipy import stats
import operator

class classifier:

    def __init__(self, k=3):
        self.k = k
        self.X = None
        self.Y = None

    #Store the train_x and train_y data
    def fit(self, X, Y):
        self.X = X
        self.Y = Y

    #test the data using the trained model
    def predict(self, X):
        hypotheses = []
        for testrow in X:
            distancelist = {}
            kitems = {}
            knearestlabels = []
            #for every row in test_X data, find the distance between the current test data and all the train_X data
            for idx, trainrow in enumerate(self.X):
                dist = distance.euclidean(testrow, trainrow)
                distancelist[idx] = dist
            sorteddistances = sorted(distancelist.items(), key=operator.itemgetter(1))
            #Pull out the k elements from the sorted distance list
            kitems = sorteddistances[:self.k]
            for kitem in kitems:
                knearestlabels.append(self.Y[kitem[0]])
            #Find the majority occurence of the k data points
            majority = int(stats.mode(knearestlabels)[0])
            hypotheses.append(majority)
        return hypotheses