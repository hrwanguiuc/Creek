from util import *
from counter import *
import math
import time


class cl:
    def __int__(self):
        self.k = 1  # smoothing value, should be changed later, default is 1

    def setValidLabels(self, valid_labels):
        self.valid_labels = valid_labels

    def setSmoothing(self, k):
        self.k = k

    def setVocabulary(self, vocabulary):
        self.vocabulary = vocabulary

    def setMethod(self, method):
        self.method = method

    def train(self, trainingData, trainingLabel):
        if self.method == 'm':
            self.multinominal_train(trainingData, trainingLabel)

        elif self.method == 'b':
            self.bernoulli_train(trainingData, trainingLabel)

    def multinominal_train(self, trainingData, trainingLabel):
        # print("this is labels: ", trainingLabel)
        # print("First tra data is: ", trainingData)

        '''
        Trains the classifier by collecting counts over the training data, and
        stores the Laplace smoothed estimates so that they can be used to classify.
        '''
        prior = Counter()  # prior count: class = 0,1
        conditional = {}  # likelihood for 1
        zero_cond = {}  # likelihood for 0
        total = dict.fromkeys((x for x in range(0, 40)), 0)  # total number of words in this class

        for word in self.vocabulary:
            conditional[word] = dict.fromkeys((x for x in range(0, 40)), 0)
            zero_cond[word] = dict.fromkeys((x for x in range(0, 40)), 0)
        for label in trainingLabel:
            prior[label] += 1
            total[label] += 1
        # print("len of tra data: ", len(trainingData))
        for i in range(len(trainingData)):

            datum = trainingData[i]  # datum is an individual topic
            # format is:
            # {'damme': 1, 'segal': 1, 'arnold': 1, 'going': 1}

            label = trainingLabel[i]  # label is the int value 0 or 1
            for word, count in datum.items():
                conditional[word][label] += 1

                # total[label] += len(datum)  # total number of words in each review

        # smoothing
        for word in self.vocabulary:
            for label in self.valid_labels:
                # print("word:", word)
                # print("label:", label)
                # print("count:", conditional[word][label])
                # print("total:", total[label])

                conditional[word][label] = \
                    (conditional[word][label] + self.k) / (total[label] + self.k * 2)  # len(self.vocabulary))
                zero_cond[word][label] = 1 - conditional[word][label]

                conditional[word][label] = math.log(conditional[word][label])
                zero_cond[word][label] = math.log(zero_cond[word][label])

        # normalize prior prob
        prior.normalize()
        self.prior = prior
        self.conditionals = conditional
        self.zero_cond = zero_cond

    def calculateLogJointProbabilities(self, datum):
        res_log = Counter()

        for label in self.valid_labels:
            res_log[label] += math.log(self.prior[label])

            for word in self.vocabulary:
                if word in datum.keys():
                    temp_log = self.conditionals[word][label]
                else:
                    temp_log = self.zero_cond[word][label]

                res_log[label] += temp_log  # math.log(temp_prob)

        return res_log

    def bernoulli_train(self, trainingData, trainingLabel):

        prior = Counter()  # prior count: class = 0,1
        conditional = {}  # likelihood
        total = dict.fromkeys((x for x in range(0, 40)), 0)

        # total number of words in different labels
        for word in self.vocabulary:
            conditional[word] = dict.fromkeys((x for x in range(0, 40)), Counter())  # {label:appearance}
        for label in trainingLabel:
            prior[label] += 1
            total[label] += 1

        for i in range(len(trainingData)):

            datum = trainingData[i]  # datum is an individual review
            # format is:
            # {'damme': 1, 'segal': 1, 'arnold': 1, 'going': 1}

            label = trainingLabel[i]  # label is the int value 0 or 1
            for word, count in datum.items():
                conditional[word][label][1] += 1

        # smoothing
        for word in self.vocabulary:
            for label in self.valid_labels:
                conditional[word][label][1] = \
                    (conditional[word][label][1] + self.k) / (total[label] + self.k * len(self.vocabulary))
                conditional[word][label][0] = 1 - conditional[word][label][1]

        # normalize prior prob
        prior.normalize()
        self.prior = prior
        self.conditionals = conditional
        # print("len of conditionals:" , len(self.conditionals))

    def calculateLogJointProbabilities_bernoulli(self, datum):
        res_log = Counter()

        for label in self.valid_labels:
            res_log[label] += math.log(self.prior[label])

            for word in self.vocabulary:
                if word in datum.keys():
                    temp_prob = (self.conditionals[word][label][1])
                else:
                    temp_prob = (self.conditionals[word][label][0])

                res_log[label] += math.log(temp_prob)

        return res_log

    def classify(self, testData):
        '''
        this is used to classify data based on the likelihood obtained: self.conditionals
        :param testData: unknown data
        :return: posterior
        '''
        res = []  # list of result
        self.posteriors = []
        if self.method == 'm':
            n = 0
            for datum in testData:
                posterior = self.calculateLogJointProbabilities(datum)
                self.posteriors.append(posterior)  # add all of the posteriors for one datum into the list
                res.append(posterior.argMax())
                print("n: ", n)
                # print("posterior: \n", posterior)
                n += 1


        elif self.method == 'b':
            n = 0
            for datum in testData:
                posterior = self.calculateLogJointProbabilities_bernoulli(datum)
                self.posteriors.append(posterior)  # add all of the posteriors for one datum into the list
                res.append(posterior.argMax())
                print("n: ", n)
                print("posterior: \n", posterior)
                n += 1

        return res

    ##################################################
    ## This part is used for odds ratio plot        ##
    ##################################################
    def odds_calculation(self, l1, l2):
        '''
        :param l1: first label
        :param l2: second label
        :return: an odd ratio list
        '''
        ratio_list = []
        if self.method == 'm':
            for word in self.vocabulary:
                p1 = self.conditionals[word][l1]
                p2 = self.conditionals[word][l2]

                ratio = p1 / p2
                ratio_list.append((word, ratio))  # A tuple (word,ratio)

        elif self.method == 'b':
            for word in self.vocabulary:
                p1 = self.conditionals[word][l1][1]
                p2 = self.conditionals[word][l2][1]

                ratio = p1 / p2
                ratio_list.append((word, ratio))  # A tuple (word,ratio)

        return self.ratio_log(ratio_list)

    def ratio_log(self, ratio_list):
        res = copy.deepcopy(ratio_list)

        for i in range(len(res)):
            res[i] = (res[i][0], math.log(res[i][1]))  # A tuple (word,log_ratio)

        res = sorted(res, key=lambda x: x[1])
        res.reverse()
        res = res[:10]
        return res

    def topHighestConditionals(self, label):
        res = []
        if self.method == 'm':
            temp = []
            for word in self.vocabulary:
                temp.append((word, self.conditionals[word][label]))  # tuple -> (word, likelihood)
            res = sorted(temp, key=lambda x: x[1])

        elif self.method == 'b':
            temp = []
            for word in self.vocabulary:
                temp.append((word, self.conditionals[word][label][1]))
            res = sorted(temp, key=lambda x: x[1])

        res.reverse()
        res = res[:10]
        return res


def test():
    trainingData, trainingLabel, vocabulary = readFile("fisher_40topic/fisher_train_40topic.txt", 10244)
    testData, testLabel, test_vocabulary = readFile("fisher_40topic/fisher_test_40topic.txt", 10)
    # initialization for the classifier
    # trainingData, trainingLabel, vocabulary = readFile("../fisher_2topic/fisher_train_2topic.txt", 878)
    # testData, testLabel, test_vocabulary = readFile("../fisher_2topic/fisher_test_2topic.txt", 98)
    # initialization for the classifier
    classifier = cl()
    classifier.setSmoothing(1)
    classifier.setVocabulary(vocabulary)
    # classifier.setValidLabels([0,1])
    classifier.setValidLabels([x for x in range(0, 40)])
    classifier.setMethod('m')
    st = time.clock()
    print("Ready to training!")
    classifier.train(trainingData, trainingLabel)
    print("Training is done!")
    prediction = classifier.classify(testData)
    accuracy = calAccuracy(prediction, testLabel, 10)
    print(prediction)

    print("-----------------------------------------------")
    print("The accuracy is: ", accuracy)
    print("Execution time is: %fs" % (time.clock() - st))
    # c_matrix = confusionMatrix(prediction, testLabel, 40, 1154)
    # outputFile(c_matrix)


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


def outputFile(c_matrix, words_class0=None, words_class1=None, odds_class0=None, odds_class1=None):
    f = open('topic_confusion_matrix_b1.txt', 'w')
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


if __name__ == "__main__":
    # pass
    test()
