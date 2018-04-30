from classifier import classifier
import pandas as pd
from scipy.io import arff
from sklearn.metrics import accuracy_score

class knn(classifier):
    def __init__(self, k=3):
        classifier.super().__init__(self,k)
    
    def main():
        #loading the dataset
        d = arff.loadarff('PhishingData.arff')
        phishingdata = pd.DataFrame(d[0])
        for col in phishingdata.columns:
            phishingdata[col] = phishingdata[col].str.decode('utf-8')
        
        #Creating train and test data by splitting the dataset into 80% and 20% respectively
        phishingdata_80percent = phishingdata.iloc[:1082,:]
        phishingdata_20percent = phishingdata.iloc[1082:,:]
        train_x = phishingdata_80percent.iloc[:,:9]
        train_y = phishingdata_80percent.iloc[:,9]
        test_x = phishingdata_20percent.iloc[:,:9]
        test_y = phishingdata_20percent.iloc[:,9]
        
        #creating an array for each row converting string to int values
        trainxvalues = []
        for trainxval in train_x.values:
            trainxarray = [int(x1) for x1 in trainxval]
            trainxvalues.append(trainxarray)
        
        trainyvalues = [int(trainyval) for trainyval in train_y.values]
        
        testxvalues = []
        for testxval in test_x.values:
            testxarray = [int(x) for x in testxval]
            testxvalues.append(testxarray)
                
        testyvalues = [int(testyval) for testyval in test_y.values]
        
        #For different values of k ranging from 2 to 32, call the classifier, fit and predict methods
        for i in range(2,33):
            c = classifier(i)
            c.fit(trainxvalues, trainyvalues)
            hyp = c.predict(testxvalues)
            print('k = %d -> Accuracy: %0.4f' %(i, accuracy_score(testyvalues, hyp)))
    
    if __name__ == '__main__':
        main()