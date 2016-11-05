from movieNavieBayes import *
import time
import matplotlib.pyplot as plt
import numpy as np


def classify():
    classifier = navieBayesClassifier()
    trainingData, trainingLabel, vocabulary = readFile("movie_review/rt-train.txt", 4000)
    testData, testLabel, test_vocabulary = readFile("movie_review/rt-test.txt", 1000)

    # initialization for the classifier
    classifier.setSmoothing(1)
    classifier.setMethod('b')
    classifier.setVocabulary(vocabulary)
    classifier.setValidLabels([0, 1])

    # start
    st = time.clock()
    print("Ready to training!")
    classifier.train(trainingData, trainingLabel)
    print("Training is done!")
    prediction = classifier.classify(testData)
    accuracy = calAccuracy(prediction, testLabel, 1000)
    c_matrix = confusionMatrix(prediction, testLabel, 2, 1000)
    write_c_matrix(c_matrix)
    print("-----------------------------------------------")
    print("The accuracy is: ", accuracy)
    print("Execution time is: %fs" % (time.clock() - st))


def calAccuracy(prediction, testLabel, n):
    '''
    calculate the accuracy of testing
    :param res:
    :param testLabel:
    :param n:
    :return:
    '''
    count = 0
    for i in range(n):
        if prediction[i] == testLabel[i]:
            count += 1

    accuracy = float(count / n)

    return accuracy


def confusionMatrix(prediction, testLabel, n, total):
    '''

    :param prediction:
    :param testLabel:
    :param n: number of labels, i.e. n = 2
    :param total: number of test data
    :return: a 2*2 matrix
    '''

    c_matrix = [[0] * n for i in range(n)]
    for i in range(total):
        c_matrix[testLabel[i]][prediction[i]] += 1

    return c_matrix


def write_c_matrix(c_matrix):
    f = open('movie_confusion_matrix_b.txt', 'w')
    f.write('HINT: row index is true value, column index is predicted value\n')
    f.write('--------------------------------------------------------------\n')
    f.write(' ' + str([0, 1]) + '\n')
    for i in range(len(c_matrix)):
        f.write(str(i) + str(c_matrix[i]) + '\n')

    rate_dict = {}
    for i in range(len(c_matrix)):
        rate = c_matrix[i][i] / sum(c_matrix[i])
        rate_dict[i] = rate

    f.write('\n--------------------------------------------------------------\n')
    f.write("Classification rate for each digit:\n")
    for i in rate_dict:
        f.write('The class ' + str(i) + ': ' + str(rate_dict[i]) + "\n")

    f.close()


def plt_cook(classifier):
    confused_label_pairs = [(1, 8), (3, 5), (4, 9), (3, 8), (7, 9)]
    logs = {}
    log_ratios = {}
    for i in confused_label_pairs:
        l1 = i[0]
        l2 = i[1]
        logs[l1] = classifier.one_odds_calculation(l1)
        logs[l2] = classifier.one_odds_calculation(l2)
        log_ratios[i] = classifier.odds_calculation(l1, l2)

    log_ratios_array = np.array(log_ratios[(3, 5)])
    log_ratios_matrix = np.reshape(log_ratios_array, (28, 28))
    test_case1 = np.array(logs[1])
    test_matrix1 = np.reshape(test_case1, (28, 28))

    plt.imshow(test_matrix1)
    plt.clim(vmin=-4, vmax=2)
    plt.colorbar()

    plt.show()


if __name__ == "__main__":
    classify()
