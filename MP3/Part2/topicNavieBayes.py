from util import *
from counter import *
import math


class navieBayesClassifier:
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
        conditional = {}  # likelihood
        total = {0: 0, 1: 0}  # total number of words in this class

        for word in self.vocabulary:
            conditional[word] = {0: 0, 1: 0}
        for label in trainingLabel:
            prior[label] += 1

        # print("len of tra data: ", len(trainingData))
        for i in range(len(trainingData)):

            datum = trainingData[i]  # datum is an individual review
            # format is:
            # {'damme': 1, 'segal': 1, 'arnold': 1, 'going': 1}

            label = trainingLabel[i]  # label is the int value 0 or 1
            for word, count in datum.items():
                conditional[word][label] += count

            total[label] += len(datum)  # total number of words in each review

        # smoothing
        for word in self.vocabulary:
            for label in self.valid_labels:
                conditional[word][label] = \
                    float(conditional[word][label] + self.k) / (total[label] + len(self.vocabulary))

        # normalize prior prob
        prior.normalize()
        self.prior = prior
        self.conditionals = conditional
        # print("len of conditionals:", len(self.conditionals))

    def calculateLogJointProbabilities(self, datum):
        res_log = Counter()

        for label in self.valid_labels:
            res_log[label] += math.log(self.prior[label])

            for word, count in datum.items():
                if word in self.conditionals.keys():
                    temp_prob = (self.conditionals[word][label]) ** (count)
                    res_log[label] += math.log(temp_prob)

        return res_log

    def bernoulli_train(self, trainingData, trainingLabel):

        prior = Counter()  # prior count: class = 0,1
        conditional = {}  # likelihood
        total = len(trainingLabel)  # total number of reviews
        for word in self.vocabulary:
            conditional[word] = {0: Counter(), 1: Counter()}  # {label:appearance}
        for label in trainingLabel:
            prior[label] += 1

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
                    float(conditional[word][label][1] + self.k) / (total + len(self.vocabulary))
                conditional[word][label][0] = 1 - conditional[word][label][1]

        # normalize prior prob
        prior.normalize()
        self.prior = prior
        self.conditionals = conditional
        # print("len of conditionals:", len(self.conditionals))

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
            for datum in testData:
                posterior = self.calculateLogJointProbabilities(datum)
                self.posteriors.append(posterior)  # add all of the posteriors for one datum into the list
                res.append(posterior.argMax())

        elif self.method == 'b':
            for datum in testData:
                posterior = self.calculateLogJointProbabilities_bernoulli(datum)
                self.posteriors.append(posterior)  # add all of the posteriors for one datum into the list
                res.append(posterior.argMax())

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
    pass


if __name__ == "__main__":
    pass
    # test()
