from topicNavieBayes import *
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet

'''
File name list:

"fisher_40topic/fisher_train_40topic.txt",10244
"fisher_40topic/fisher_test_40topic.txt",1154
'''


def classify():
    classifier = navieBayesClassifier()
    # trainingData, trainingLabel, vocabulary = readFile("movie_review/rt-train.txt",4000)
    # testData, testLabel, test_vocabulary = readFile("movie_review/rt-test.txt",1000)

    trainingData, trainingLabel, vocabulary = readFile("fisher_40topic/fisher_train_40topic.txt", 10244)
    # trainingData, trainingLabel, vocabulary = lemma_readFile("fisher_40topic/fisher_train_40topic.txt",10244)
    testData, testLabel, test_vocabulary = readFile("fisher_40topic/fisher_test_40topic.txt", 1154)
    # initialization for the classifier
    vocabulary = lemmatization(vocabulary)
    classifier.setSmoothing(1)  # set different smoothing value
    classifier.setMethod('m')
    classifier.setVocabulary(vocabulary)
    classifier.setValidLabels([x for x in range(0, 40)])

    # start
    st = time.clock()
    print("Ready to training!")
    classifier.train(trainingData, trainingLabel)
    print("Training is done!")
    prediction = classifier.classify(testData)
    accuracy = calAccuracy(prediction, testLabel, 1154)
    print(prediction)
    c_matrix = confusionMatrix(prediction, testLabel, 40, 1154)

    # get the top 10 words with highest likelihood
    # words_class0 = classifier.topHighestConditionals(0)
    # words_class1 = classifier.topHighestConditionals(1)
    # get the top 10 words with highest odds ratio
    # odds_class0 = classifier.odds_calculation(0, 1)
    # odds_class1 = classifier.odds_calculation(1, 0)

    most = most_likely_confused(c_matrix)
    print("Most likely confused topic: ", most)
    # cm_plot(c_matrix)
    # outputFile(c_matrix,most)
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


def outputFile(c_matrix, most_likely_confused, words_class0=None, words_class1=None, odds_class0=None,
               odds_class1=None):
    f = open('topic_confusion_matrix_b5.txt', 'w')
    f.write('HINT: row index is actual value, column index is predicted value\n')
    f.write('--------------------------------------------------------------\n')
    f.write(' ' + str([x for x in range(0, 40)]) + '\n')
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
    f.write('\n--------------------------------------------------------------\n')
    f.write("Most likely confused topic pairs for each individual topic:\n")
    for t1, t2 in most_likely_confused.items():
        f.write("topic " + str(t1) + ":" + str(t2) + '\n')

    '''
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
    '''
    f.close()


def most_likely_confused(matrix):
    res = {}
    for i in range(len(matrix)):
        second_max = sorted(matrix[i])[-2]
        for j in range(len(matrix[i])):
            if matrix[i][j] == second_max:
                res[i] = j

    return res


def cm_plot(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            matrix[i][j] = round(matrix[i][j], 2)

    new_matrix = np.array(matrix)

    df_cm = pd.DataFrame(new_matrix, index=[i for i in [str(x) for x in range(40)]],
                         columns=[i for i in [str(x) for x in range(40)]])
    fig, ax = plt.subplots(figsize=(25, 15))  # Sample figsize in inches
    sb.heatmap(df_cm, annot=True, linewidths=.5, ax=ax)
    # sb.heatmap(df_cm, annot=True)
    plt.savefig("cm.png")
    plt.show()


def lemmatization(vocabulary):
    wordnet_lemmatizer = WordNetLemmatizer()
    # convert to lower case
    vocabulary = map(str.lower, list(vocabulary))
    # remove stop words
    stop_words = set(stopwords.words('english'))
    for i in ['eh', 'uh', 'um', 'know']:
        stop_words.add(i)
    print(stop_words)
    new_vocabulary = [w for w in vocabulary if not w in stop_words]
    # assign pos_tags
    new_vocabulary = nltk.pos_tag(new_vocabulary)
    # lemmatize
    print(new_vocabulary)
    print(len(new_vocabulary))
    for i in range(len(new_vocabulary)):
        new_vocabulary[i] = wordnet_lemmatizer.lemmatize(str(new_vocabulary[i][0]),
                                                         get_wordnet_pos(new_vocabulary[i][1]))
    new_vocabulary = set(new_vocabulary)
    print(len(new_vocabulary))

    return new_vocabulary


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


if __name__ == "__main__":
    # trainingData, trainingLabel, vocabulary = readFile("fisher_40topic/fisher_train_40topic.txt",10244)
    # lemmatization(vocabulary)
    classify()
