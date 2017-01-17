from util import *
import numpy as np
import time
import math
import execution

TRAINING_LEN = 5000
TEST_LEN = 1000
IMG_WIDTH = 28
IMG_HEIGHT = 28


class knnClassifier:
    def setValidLabels(self, validLabels):
        self.validLabels = validLabels

    def setK(self, k):
        self.k = k

    # this function will return Hamming distance as the similarity measure
    def distance(self, testData, trainData):
        res = sum(testData ^ trainData)
        return res

    # this function will return the euclidean distance as the similarity measure
    def euclidean(self, testData, trainData):
        # return sum(math.sqrt((testData-trainData) ** 2))
        return sum(map(math.sqrt, (testData - trainData) ** 2))
        # cos = np.dot(testData,trainData)/(sum(testData)*sum(trainData))
        # return cos

    def vote(self, k_idx):
        temp = dict.fromkeys((x for x in range(10)), 0)
        for i in k_idx:
            label = self.trainingLabel[i]
            temp[label] += 1

        return sorted(temp.items(), key=lambda x: x[1], reverse=True)[0][0]

    # training is just used to load the training data and training labels
    def train(self, trainingData, trainingLabel):
        self.trainingData = trainingData
        self.trainingLabel = trainingLabel

    def classify(self, testData, testLabel):

        res = []

        for i in range(len(testData)):
            datum = testData[i]
            neighbors_distance = {}
            '''
            for j in range(len(self.trainingData)):
                # calculate distance between an individual testData and all trainingData
                neighbors_distance[j] = self.distance(datum,self.trainingData[j])
                #neighbors_distance[j] = self.euclidean(datum, self.trainingData[j])

            nearest_neighbors = sorted(neighbors_distance.items(),
                                       key = lambda x: x[1], reverse = False)[:self.k]
            k_idx = []
            for k in nearest_neighbors:
                k_idx.append(k[0])

            prediction = self.vote(k_idx)
            res.append(prediction)

            '''
            temp = np.dot(self.trainingData, datum)
            k_idx = np.argsort(temp)[-self.k:]
            # find labels of k biggest among 5000 distance
            prediction = self.vote(k_idx)
            res.append(prediction)

            print("i: ", i)

        return self.result(res, testLabel)

    def result(self, res, testLabel):
        confusion_matrix = np.zeros((10, 10))
        correct_labels = 0
        for i in range(len(testLabel)):
            if res[i] == testLabel[i]:
                correct_labels += 1
            confusion_matrix[testLabel[i], res[i]] += 1

        accuracy = correct_labels / len(testLabel)
        return accuracy, confusion_matrix


def main():
    classifier = knnClassifier()
    start = time.clock()

    trainingData = knn_readDataFile("digitdata/trainingimages", 5000)
    trainingLabel = readLabelFile("digitdata/traininglabels", 5000)

    testData = knn_readDataFile("digitdata/testimages", 1000)
    testLabel = readLabelFile("digitdata/testlabels", 1000)

    classifier.setValidLabels([x for x in range(10)])
    classifier.setK(4)  # k-NN

    print("Ready for training.")
    classifier.train(trainingData, trainingLabel)
    print("Training is done.")

    accuracy, confusion_matrix = classifier.classify(testData, testLabel)
    print("classification is done.")
    print("-----------------------------------------------------------")
    print("The accuracy is: ", accuracy)
    print("Execution time is: %fs" % (time.clock() - start))
    print(confusion_matrix)
    # execution.cm_plot(confusion_matrix)


if __name__ == "__main__":
    main()
