from util import *
from counter import *
import math
import itertools

class navieBayesClassifier2:

    def __int__(self, valid_labels):
        self.valid_labels = valid_labels
        self.k = 1 # smoothing value, should be changed later, default is 1


    def setSmoothing(self,k):

        self.k = k


    def train(self, trainingData, trainingLabel):
        #print("this is labels: ", trainingLabel)
        #print("First tra data is: ", trainingData)


        self.initFeatures(self.row, self.col, self.overlap) # feature dictionary initialization
        self.initValues(self.row,self.col) # init value list: e.g. '0000', '0001'

        '''
        Trains the classifier by collecting counts over the training data, and
        stores the Laplace smoothed estimates so that they can be used to classify.
        '''
        prior = Counter()  # prior count: class = 0, 1, 2 ... 9
        conditional = {}  # general count, with F(i,j) = 0 or F(i,j) = 1, used for conditional prob
        total = Counter() # total number of training examples from this class

        for pixel in self.features:
            conditional[pixel] = {}
            for value in self.valueList:
                conditional[pixel][value] = Counter()

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
            for value in self.valueList:
                for label in self.valid_labels:
                    modified_conditional[pixel][value][label] = \
                        (conditional[pixel][value][label]+self.k)/(total[label]+2**(self.row*self.col)*self.k)

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



    def setBoxSize(self, row, col, overlap):

        self.row = row
        self.col = col
        self.overlap = overlap

    def initFeatures(self, row, col,overlap = False):
        self.features = []
        if overlap:
            for i in range(IMG_HEIGHT - row +1):
                for j in range(IMG_WIDTH - col + 1):
                    temp = []
                    for r in range(row):
                        for c in range(col):
                            temp.append((i+r,j+c))
                    self.features.append(str(temp))
        else:
            i,j = 0,0
            while i < IMG_HEIGHT:
                while j < IMG_WIDTH:
                    temp=[]
                    for r in range(row):
                        for c in range(col):
                            temp.append((i + r, j + c))
                    self.features.append(str(temp))
                    j += col
                i += row
                j = 0


    def initValues(self,row,col):
        res = ["".join(seq) for seq in itertools.product("01", repeat= (row*col))]
        self.valueList = res





# construct trainingData:
# trainingData = map(featureConstruction, raw_trainingData)
def featureConstruction(Gfeatures, datum):
    '''

    :param datum: Datum class
    :return: a list of feature

    feature is a dictionary indicates whether the value is 0 or 1
    '''

    features = Counter()

    for G in Gfeatures:
        temp_str = ''
        for i in range(len(G)):
            temp_int = datum.getPixel(G[i][0],G[i][1])
            temp_str += str(temp_int)

        features[str(G)] = temp_str

    return features


def initG(row, col, overlap=False):
    Gfeatures = []
    if overlap:
        for i in range(IMG_HEIGHT - row + 1):
            for j in range(IMG_WIDTH - col + 1):
                temp = []
                for r in range(row):
                    for c in range(col):
                        temp.append((i + r, j + c))
                Gfeatures.append(temp)
    else:
        i, j = 0, 0
        while i < IMG_HEIGHT:
            while j < IMG_WIDTH:
                temp = []
                for r in range(row):
                    for c in range(col):
                        temp.append((i + r, j + c))
                Gfeatures.append(temp)
                j += col
            i += row
            j = 0

    return Gfeatures


def train(trainingData, trainingLabel):
    pass




def test():
    raw_trainingData = readDataFile("digitdata/trainingimages",1)
    Gfeatures = initG(2,2,True)
    trainingData=[]
    for i in range(len(raw_trainingData)):
        temp_features = featureConstruction(Gfeatures,raw_trainingData[i])
        trainingData.append(temp_features)
    print("TD:",len(trainingData[0]))
    cl = navieBayesClassifier2()
    cl.initFeatures(2,2,True)
    print(len(cl.features))
    print(cl.features[0])
    print(cl.features[1])
    print(cl.features[7])
    for i in range(28):
        temp = ''
        for j in range(28):
            temp += str(raw_trainingData[0].getPixel(i, j))
        print(temp)
    cl.initValues(2, 4)
    print(len(cl.valueList))


if __name__ == "__main__":
    pass
    #test()