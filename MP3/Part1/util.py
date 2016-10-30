
from functools import *
import copy
##################################################################
# This is designed as a fundation for the navie bayes classifer  #
##################################################################


IMG_WIDTH = 28
IMG_HEIGHT = 28

class Datum:
    # pixels for an image

    #28 # # # #      #  #
    #27 # # # #      #  #
    # .
    # .
    # .
    # 3 # # + #      #  #
    # 2 # # # #      #  #
    # 1 # # # #      #  #
    # 0 # # # #      #  #
    #   0 1 2 3 ... 27 28

    def __init__(self, data, width, height):

        #IMG_WIDTH = width
        #IMG_HEIGHT = height
        self.width = IMG_WIDTH
        self.height = IMG_HEIGHT
        if data == None:
            data = [[' ' for i in range(IMG_WIDTH)] for j in range(IMG_HEIGHT)]
        #self.pixel = matrix_invert(convert2Integer(data))
        self.pixel = convert2Integer(data)
        self.data = data

    def getPixel(self, row, col):

        return self.pixel[row][col]

    def getAllPixels(self,row,col):

        return self.pixel
    def toStr(self,data):
        for i in range(len(data)):
            temp_str = ''
            for j in range(len(data[i])):
                temp_str += data[i][j]
            print(temp_str)

    def __str__(self):
       return self.toStr(self.data)



def matrix_invert(array):

    res = [[] for i in array]
    for outer in array:
        for inner in range(len(outer)):
            res[inner].append(outer[inner])
    
    return res


# n is the number of training/testing data
def readDataFile(filename, n):
    '''
    read n data from file and return Datum obj
    '''
    width = IMG_WIDTH
    height = IMG_HEIGHT

    f = open(filename,'r')

    lines = f.readlines()
    for i in range(len(lines)):
        lines[i] = lines[i].rstrip('\n') # remove '\n' in the line
    lines.reverse() # reverse the list in order to use the pop() method to get the first element
    res_list = []
    for i in range(n):
        temp_data = []
        for r in range(height):
            temp_data.append(list(lines.pop())) # get the line from lines, and add it to the temp_data, totally 28 for one temp_data
        # control the situation when going to the end of file
        if len(temp_data[0]) < (width - 1):
            break
        # create a datum due to temp_data
        res_list.append(Datum(temp_data,width,height))

    f.close()
    # res_list should be a list containing 5000 training datums.
    return res_list


def readLabelFile(filename,n):

    '''
    :param filename: label filename
    :param n: number of labels
    :return: list of labels
    '''
    f = open(filename,'r')
    lines = f.readlines()
    label_list = []
    for i in range(n):

        label_list.append(int(lines[i]))

    return label_list


def convert2Integer(data):
    temp_data = copy.deepcopy(data)
    for i in range(len(data)):
        for j in range(len(data[i])):
            temp_data[i][j] = parseChar(temp_data[i][j])
    #print("temp_data: ", len(temp_data))
    return temp_data



def parseChar(ch):

    if ch == ' ':
        return 0
    else:
        return 1





def test():
    training_data = readDataFile("digitdata/trainingimages",1)
    training_label = readLabelFile("digitdata/traininglabels",100)
    print("labels: ",training_label)
    print("length of training data is: ",len(training_data))
    for i in range(1):
        datum = training_data[i]
        datum.toStr(datum.data)
        pixel = datum.getAllPixels(28,28)
        print("pixel: ", len(pixel))
        for i in range(28):
            temp =''
            for j in range(28):
                temp += str(datum.getPixel(i,j))
            print(temp)




if __name__ == "__main__":

    pass
    #test()


