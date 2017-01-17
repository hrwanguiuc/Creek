import copy
import pandas as pd
from counter import *
import topicExecution
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.corpus import stopwords


def readFile(filename, n):
    '''
    read n data from file and return Datum obj
    '''

    f = open(filename, 'r')

    lines = f.readlines()
    for i in range(len(lines)):
        lines[i] = lines[i].rstrip('\n')  # remove '\n' in the line

    label_list = []
    review_list = []
    vocabulary = set()
    for i in range(n):
        rm_space = lines[i].split(sep=' ')
        temp_num = int(rm_space.pop(0))
        if temp_num == -1:
            temp_num = 0
        label_list.append(temp_num)
        word_dict = {}
        for j in range(len(rm_space)):
            word_count = rm_space[j].split(":")
            word = word_count[0]
            vocabulary.add(word)
            count = word_count[1]
            word_dict[word] = int(count)
        review_list.append(word_dict)

    f.close()
    # res_list should be a list containing 5000 training datums.
    return review_list, label_list, vocabulary


class Datum:
    '''
    Datum is an individual bag of words counting for movie review
    '''

    def __init__(self, data):
        self.data = data


def lemma_readFile(filename, n):
    stop_words = set(stopwords.words('english'))
    wordnet_lemmatizer = WordNetLemmatizer()

    f = open(filename, 'r')

    lines = f.readlines()
    for i in range(len(lines)):
        lines[i] = lines[i].rstrip('\n')  # remove '\n' in the line

    label_list = []
    review_list = []
    vocabulary = set()
    for i in range(n):
        rm_space = lines[i].split(sep=' ')
        temp_num = int(rm_space.pop(0))
        if temp_num == -1:
            temp_num = 0
        label_list.append(temp_num)
        word_dict = {}
        for j in range(len(rm_space)):
            word_count = rm_space[j].split(":")
            word = word_count[0]
            if word in stop_words:
                continue
            temp = [word]
            temp = nltk.pos_tag(temp)
            word = temp[0]
            word = wordnet_lemmatizer.lemmatize(str(word[0]), topicExecution.get_wordnet_pos(word[1]))

            vocabulary.add(word)
            count = word_count[1]
            word_dict[word] = int(count)
        review_list.append(word_dict)

    f.close()

    print("len of voc:", len(vocabulary))

    # res_list should be a list containing 5000 training datums.
    return review_list, label_list, vocabulary




def test():
    filename = "movie_review/rt-test.txt"
    # trainingData, traininglabel, vocabularty = readFile(filename, 1000)
    trainingData, trainingLabel, vocabulary = lemma_readFile("fisher_40topic/fisher_train_40topic.txt", 10244)

    print("Raw training data: ", trainingData[0])
    print("training label: ", trainingLabel[0])
    print("size of raw data: ", len(trainingData))
    print("size of label: ", len(trainingLabel))
    print("size of vocabulary: ", len(vocabulary))
    #

if __name__ == "__main__":
    test()
