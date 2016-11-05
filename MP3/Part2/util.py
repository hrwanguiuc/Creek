import copy
import pandas as pd
from counter import *


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


def test():
    filename = "movie_review/rt-test.txt"
    trainingData, traininglabel, vocabularty = readFile(filename, 1000)
    print("Raw training data: ", trainingData[999])
    print("training label: ", traininglabel[999])
    print("size of raw data: ", len(trainingData))
    print("size of label: ", len(traininglabel))
    print("size of vocabulary: ", len(vocabularty))


if __name__ == "__main__":
    test()
