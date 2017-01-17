from util import *
import numpy as np
import matplotlib.pyplot as plt

EPOCH = 100
TRAINING_LEN = 5000
TEST_LEN = 1000
IMG_WIDTH = 28
IMG_HEIGHT = 28


class perceptron:
    def __init__(self):
        self.perceptrons = np.zeros((10, IMG_HEIGHT * IMG_WIDTH + 1))

    def setEpoch(self, epoch):
        self.epoch = epoch

    # this is for the ordering of the training examples
    def setRandom(self, random=False):
        self.random = random

    def setDecay(self, decay):
        self.decay = decay

    # this is for the initialization of weights
    def setRandomize(self, randomize=False):
        self.randomize = randomize
        # init perceptrons in a random number between 0 and 1
        if self.randomize:
            np.random.seed(1)
            self.perceptrons = np.random.rand(10, IMG_HEIGHT * IMG_WIDTH + 1)

    def train(self, trainingData, trainingLabel):
        np.random.seed(1)

        # Several things to remember in each epoch [for loop]:
        # 1. alpha = f(decay);  2. init num_correct_labels = 5000;

        self.accuracy_epoch = np.zeros(self.epoch)
        # empty array for recording the training accuracy of each epoch

        if self.random:
            for epoch in range(self.epoch):
                alpha = self.decay / (self.decay + epoch)
                num_correct_labels = len(trainingData)

                for i in np.random.permutation(len(trainingData)):
                    datum = trainingData[i]
                    label = trainingLabel[i]

                    res = np.zeros(10)
                    for j in range(10):
                        res[j] = np.dot(self.perceptrons[j], datum)

                    prediction = np.argmax(res)

                    if prediction != label:
                        self.perceptrons[label] += alpha * datum
                        self.perceptrons[prediction] -= alpha * datum
                        num_correct_labels -= 1

                self.accuracy_epoch[epoch] = num_correct_labels / TRAINING_LEN
                print("# correct: ", num_correct_labels)

        else:
            for epoch in range(self.epoch):
                alpha = self.decay / (self.decay + epoch)
                num_correct_labels = len(trainingData)

                for i in range(len(trainingData)):
                    datum = trainingData[i]
                    label = trainingLabel[i]

                    res = np.zeros(10)
                    for j in range(10):
                        res[j] = np.dot(self.perceptrons[j], datum)

                    prediction = np.argmax(res)

                    if prediction != label:
                        self.perceptrons[label] += alpha * datum
                        self.perceptrons[prediction] -= alpha * datum
                        num_correct_labels -= 1

                self.accuracy_epoch[epoch] = num_correct_labels / TRAINING_LEN
                print("# correct: ", num_correct_labels)

    def classify(self, testingData, testingLabel):

        num_correct_labels = 0
        confusion_matrix = np.zeros((10, 10))

        for i in range(len(testingData)):
            datum = testingData[i]
            label = testingLabel[i]

            res = np.zeros(10)
            for j in range(10):
                res[j] = np.dot(self.perceptrons[j], datum)
            prediction = np.argmax(res)

            if prediction == label:
                num_correct_labels += 1

            confusion_matrix[label, prediction] += 1

        accuracy = num_correct_labels / len(testingLabel)

        return accuracy, confusion_matrix

    def training_curve_plot(self):
        plt.plot(self.accuracy_epoch)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.suptitle("Training Curve")
        plt.show()
