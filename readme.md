k-Nearest Neighbors classifier predicts the test instance by searching through the training set for the k most similar instances.

* The fit function accepts a list of (training) features and their labels as parameters, changing the internal state of the classifier as needed.

* The predict function accepts a list of (test) features and outputs the classifier’s hypothesis, in order, about the class membership of the instance represented by the features.