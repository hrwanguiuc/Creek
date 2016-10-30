from util import *
from counter import *
import math

class navieBayesClassifier:

    def __int__(self, valid_labels):
        self.valid_labels = valid_labels
        self.k = 1 # smoothing value, should be changed later, default is 1


    def setSmoothing(self,k):

        self.k = k


    def train(self, trainingData, trainingLabel):
        #print("this is labels: ", trainingLabel)
        #print("First tra data is: ", trainingData)
        self.initFeatures() # feature dictionary initialization

        '''
        Trains the classifier by collecting counts over the training data, and
        stores the Laplace smoothed estimates so that they can be used to classify.
        '''
        prior = Counter()  # prior count: class = 0, 1, 2 ... 9
        conditional = {}  # general count, with F(i,j) = 0 or F(i,j) = 1, used for conditional prob
        total = Counter() # total number of training examples from this class

        for pixel in self.features:
            conditional[pixel] = {0:Counter(),1:Counter()}
        for label in trainingLabel:
            total[label] += 1
            prior[label] += 1

        #print("len of tra data: ", len(trainingData))
        for i in range(len(trainingData)):

            datum = trainingData[i]
            label = trainingLabel[i]
            for pixel, value in datum.items():
                conditional[pixel][value][label] += 1

        # smoothing
        modified_conditional = copy.deepcopy(conditional)
        for pixel in self.features:
            for value in [0,1]:
                for label in self.valid_labels:
                    modified_conditional[pixel][value][label] = \
                        (conditional[pixel][value][label]+self.k)/(total[label]+2*self.k)

        # normalize prior prob
        prior.normalize()
        self.prior = prior
        self.conditionals = modified_conditional


    def calculateLogJointProbabilities(self, datum):
        res_log = Counter()

        for label in self.valid_labels:
            res_log[label] += math.log(self.prior[label])

            for i in self.conditionals:
                temp_prob = self.conditionals[i][datum[i]][label]
                res_log[label] += math.log(temp_prob)

        return res_log



    def classify(self,testData):
        '''
        this is used to classify data based on the likelihood obtained: self.conditionals
        :param testData: unknown data
        :return: posterior
        '''
        res = [] # list of result
        self.posteriors = []
        for datum in testData:
            posterior = self.calculateLogJointProbabilities(datum)
            self.posteriors.append(posterior) # add all of the posteriors for one datum into the list
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
        for pixel in self.features:
            p1 = self.conditionals[pixel][1][l1]
            p2 = self.conditionals[pixel][1][l2]

            ratio = p1/p2
            ratio_list.append(ratio)
        # ratio_list has 28*28 elements
        return self.ratio_log(ratio_list)


    def one_odds_calculation(self,label):
        '''
        this is used to calculate the odds for one label, where value = 1
        '''
        res_list = []
        for pixel in self.features:
            p = self.conditionals[pixel][1][label]
            res_list.append(p)

        return res_list


    def ratio_log(self,ratio_list):
        res = copy.deepcopy(ratio_list)

        for i in range(len(res)):
            res[i] = math.log(res[i])

        return res


    def initFeatures(self):
        self.features = []
        for i in range(IMG_WIDTH):
            for j in range(IMG_HEIGHT):
                self.features.append((i,j))







# construct trainingData:
# trainingData = map(featureConstruction, raw_trainingData)
def featureConstruction(datum):
    '''

    :param datum: Datum class
    :return: a list of feature

    feature is a dictionary indicates whether the value is 0 or 1
    '''

    features = Counter()
    for i in range(IMG_WIDTH):
        for j in range(IMG_HEIGHT):
            if datum.getPixel(i,j)>0:
                features[(i,j)] = 1
            else:
                features[(i,j)] = 0

    return features


def train(trainingData, trainingLabel):
    pass




def test():
    raw_trainingData = readDataFile("digitdata/trainingimages",1)
    trainingData = map(featureConstruction, raw_trainingData)
    cl = navieBayesClassifier()
    cl.train(trainingData,0)


if __name__ == "__main__":
    pass
    #test()