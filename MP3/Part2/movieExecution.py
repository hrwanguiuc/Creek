from movieNavieBayes import *
import time
import matplotlib.pyplot as plt
import numpy as np

'''
File name list:
"movie_review/rt-train.txt",4000
"movie_review/rt-test.txt",1000

"fisher_2topic/fisher_train_2topic.txt",878
"fisher_2topic/fisher_test_2topic.txt",98
'''

def classify():
    classifier = navieBayesClassifier()
    # trainingData, trainingLabel, vocabulary = readFile("movie_review/rt-train.txt",4000)
    # testData, testLabel, test_vocabulary = readFile("movie_review/rt-test.txt",1000)
    trainingData, trainingLabel, vocabulary = readFile("fisher_2topic/fisher_train_2topic.txt", 878)
    testData, testLabel, test_vocabulary = readFile("fisher_2topic/fisher_test_2topic.txt", 98)
    # initialization for the classifier
    classifier.setSmoothing(1)
    classifier.setMethod('m')
    classifier.setVocabulary(vocabulary)
    classifier.setValidLabels([0, 1])

    # start
    st = time.clock()
    print("Ready to training!")
    classifier.train(trainingData, trainingLabel)
    print("Training is done!")
    prediction = classifier.classify(testData)
    accuracy = calAccuracy(prediction, testLabel, 98)
    c_matrix = confusionMatrix(prediction, testLabel, 2, 98)

    # get the top 10 words with highest likelihood
    words_class0 = classifier.topHighestConditionals(0)
    words_class1 = classifier.topHighestConditionals(1)
    # get the top 10 words with highest odds ratio
    odds_class0 = classifier.odds_calculation(0, 1)
    odds_class1 = classifier.odds_calculation(1, 0)

    outputFile(c_matrix, words_class0, words_class1, odds_class0, odds_class1)
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

    accuracy = float(count) / n

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

    for i in range(len(c_matrix)):
        temp_sum = sum(c_matrix[i])
        for j in range(len(c_matrix[i])):
            c_matrix[i][j] = float(c_matrix[i][j]) / temp_sum

    return c_matrix


def outputFile(c_matrix, words_class0, words_class1, odds_class0, odds_class1):
    f = open('conversation_confusion_matrix_m.txt', 'w')
    f.write('HINT: row index is actual value, column index is predicted value\n')
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
    # highest 10 words -> likelihood
    f.write('--------------------------------------------------------------\n')
    f.write("The top 10 words with the highest likelihood for class 0:\n")
    for i in words_class0:
        f.write(i[0] + ": " + str(i[1]) + "\n")
    f.write("\nThe top 10 words with the highest likelihood for class 1:\n")
    for i in words_class1:
        f.write(i[0] + ": " + str(i[1]) + "\n")
    # highest 10 words -> odds_ratio
    f.write('--------------------------------------------------------------\n')
    f.write("\nThe top 10 words with the highest odds ratio for class 0:\n")
    for i in odds_class0:
        f.write(i[0] + ": " + str(i[1]) + "\n")
    f.write("\nThe top 10 words with the highest odds ratio for class 1:\n")
    for i in odds_class1:
        f.write(i[0] + ": " + str(i[1]) + "\n")

    f.close()



if __name__ == "__main__":
    classify()
