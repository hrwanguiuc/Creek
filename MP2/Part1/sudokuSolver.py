import copy
from functools import *


# read the sudoku grid from the file and convert it into a matrix
def gen_matrix(filename):

    with open(str(filename),'r') as f:
        lines = f.readlines()
        grid = []
        for i in range(len(lines)):
            grid.append(list(lines[i].strip()))
    return grid


def get_wordbank(filename):

    with open(str(filename),'r') as f:
        lines = f.readlines()
    return [i.strip() for i in lines]


def get_letters(wordbank):
    letterlist = set()
    str_list=''
    for i in range(len(wordbank)):
        for j in range(len(wordbank[i])):
            letterlist.add(wordbank[i][j])
    for i in letterlist:
        str_list += i

    return str_list


# return a tuple of elements in A and B
def cross(A, B):
    temp =[]
    for i in A:
        for j in B:
            temp.append((int(i),int(j)))
    return temp





#############################################
#    Initialization of variables            #
#############################################
letters = get_letters(get_wordbank('bank1.txt'))
rows = '123456789'
cols = '123456789'
# constraints for the square
squares  = cross(rows, cols) # squares is a tuple e.g. (1,1)
colConst = [cross(rows,c) for c in cols]
rowConst = [cross(r,cols) for r in rows]
gridConst = [cross(row,col) for row in ('123','456','789') for col in ('123','456','789')]
constraints = (rowConst + colConst +gridConst)



def gen_unitConst():
    temp={}
    for k in squares:
        vList=[]
        for v in constraints:
            if k in v:
                vList.append(v)
        temp[k] = vList
    return temp


# get peers for each unit
def gen_peers():
    temp={}
    for k in squares:
        tempSet =set()
        tempList = reduce(lambda x, y: x + y, units[k], [])
        for i in tempList:
            tempSet.add(i)
        tempSet.remove(k)
        temp[k]=tempSet
    return temp

units = gen_unitConst()
peers = gen_peers()

########################################################
## Function to parse the grid and manipulations       ##
########################################################


# parse the input grid and output all the possible values according to the initial state of the grid
def parse_grid(grid):

    # we assign each character in letters to each square in the grid.
    # values stands for the dictionary of the current grid
    values = {}
    result = None
    for i in squares:
        values[i]=letters  #initialization of values
    cur_grid_dict = current_grid(grid).items()
    for k, v in cur_grid_dict:
        if v in letters:
            result = assign(values, k, v)
        if result is False:
            return False
    return values # values is a dictionary


def current_grid(grid):
    templist=[]
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            # tempList is used to get the value of each unit in the grid,from (0,0)->(0,8)->(1,0)->(1,8)->...->(8,8)
            templist.append(grid[i][j])

    tempDict = dict(zip(squares, templist))
    return tempDict


def assign(values,k,v):
    if v not in values[k] and len(values[k]) == 1: # the situation when the letter overlaps, 'B' -> 'B'
        return False
    other_values = values[k].replace(v,'') # to create other values which does not include v
    for i in other_values:
        if not eliminate(values,k,i):
            return False
    return values

# eliminate v from ohter_values[k]
# k stands for key, v stands for value


def eliminate(values, k, v):

    if v not in values[k]:
        return values
    values[k] = values[k].replace(v , '')
    if len(values[k]) == 0:  # contradiction
        return False
    elif len(values[k]) == 1:  # if the possible value for values[k] is only 1, then eliminate this value in its peers
        new_v = values[k]
        if not all(eliminate(values,new_k,new_v) for new_k in peers[k]):
            return False

    return values


def grid2values(grid):
    return parse_grid(grid)


####################################
## Function for assigning word    ##
####################################

def check_length(next,word,w_index):
    if len(word)-w_index-1 <= 9-next[1] and w_index <= next[1]-1:
        return 'H'
    if len(word)-w_index-1 <= 9-next[0] and w_index <= next[0]-1:
        return 'V'

    return False


def check_index(wordbank,values,next):

    print(wordbank)
    print("next is: ", next)
    for i in range(len(wordbank)):
        for j in range(len(wordbank[i])):
            if (wordbank[i][j]==values[next]):
                temp_index = j
                word = wordbank[i]
                return (temp_index,word) # temp_index is the index of letter in the word
    return False

def findword(wordbank,values,next):

    temp_tuple = check_index(wordbank,values, next)
    if not temp_tuple:
        return False
    temp_index = temp_tuple[0]
    word = temp_tuple[1]

    result = check_length(next,word,temp_index)
    print("Order of check_length: ",result)
    if not result:
        new_bank = copy.deepcopy(wordbank)
        new_bank.remove(word)
        if len(new_bank)==0:
            return False
        else:
            return findword(new_bank, values, next)
    if result:
        if not word_assign(values,word,next,temp_index,result):
            print("not word_assign")
            wordbank.remove(word)
            return findword(wordbank,values,next)
        order = result
        return word, temp_index, order
        #return # word,, order'''


def word_assign(values,word,next,temp_index,order):
    new_values = copy.deepcopy(values)
    if order is 'H':
        upper = next[1] + len(word) - temp_index - 1
        lower = next[1] - temp_index - 1

        if all(assign(new_values, (next[0], i + 1), word[i]) for i in range(lower, upper)):
            return new_values


    if order is 'V':
        upper = next[0] + len(word) - temp_index - 1
        lower = next[0] - temp_index - 1
        # new_values = copy.deepcopy(values)
        if all(assign(new_values, (i + 1, next[1]), word[i]) for i in range(lower, upper)):
            return new_values
    return False


def dfs_search(values, wordbank,visited):
    print("Visited: ",visited)
    if values is False:
        return False
    if all(len(values[s]) == 1 for s in squares):
        return values  # In this situation, each square has the only value, which means it is solved

    next = (1, 1)  # initialization

    if next in visited:
        temp = next
        next = (temp[0],temp[1]+1)
    minvalue = len(values[next])  # minimum remaining value is the heuristic here
    for s in squares:
        if s in visited:
            continue
        #print(len(values[s]))
        if 1 <= len(values[s]) < minvalue:
            minvalue = len(values[s])
            next = s # now we get the which variable we would assign value to
    visited.add(next)

    # In this part, we should deal with the selection of one word from word bank
    # and then assign different values in the grid
    w_index = 0
    result = findword(wordbank,values,next)
    if not result:
        return dfs_search(values,wordbank,visited)
    word = result[0]
    temp_index = result[1] # index of letter in the word
    order = result[2]
    print ("The selected word is: ", word)
    print ("Order is: ",order)
    new_values = word_assign(values, word, next, temp_index, order)
    if not new_values:
        new_wordbank = copy.deepcopy(wordbank)
        new_wordbank.remove(word)
        return dfs_search(values,new_wordbank,visited)
    new_wordbank = copy.deepcopy(wordbank)
    new_wordbank.remove(word)

    print(values[(1,1)])
    return dfs_search(new_values, new_wordbank,visited)


def test():
    grid = gen_matrix('grid1.txt')
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i][j] != '_':
                grid[i][j] = grid[i][j].lower()
    wordbank = get_wordbank('bank1.txt')

    values = parse_grid(grid)

    #result = findword(wordbank,values,(6,3))
    #print(values)
    minvalue = len(values[(1, 1)])  # minimum remaining value is the heuristic here
    next = (1, 1)
    for s in squares:
        # print(len(values[s]))
        if 1 <= len(values[s]) < minvalue:
            minvalue = len(values[s])
            next = s  # now we get the which variable we would assign value to
    print(next)
    print(values[next])
    result=findword(wordbank,values,next)
    print("This is the result from findword: ",result)
    word = result[0]
    temp_index = result[1]
    order = result[2]
    print("This is word: ", word)
    print("This is temp_index: ", temp_index)
    print("This is order: ", order)

    val = word_assign(values, word, next, temp_index, order)
    print(val[1,3])
    print(val[1,2])
    print(val[2,2])
    new_wordbank = copy.deepcopy(wordbank)
    new_wordbank.remove(word)
    result2 = findword(new_wordbank, val, (1,2))
    print("This is the result from findword: ",result2)
    print(val)




def main():
    grid = gen_matrix('grid1.txt')
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i][j] != '_':
                grid[i][j] = grid[i][j].lower()
    wordbank = get_wordbank('bank1.txt')
    values = grid2values(grid)
    #print (values)
    visited = set()
    result = dfs_search(values,wordbank,visited)
    print(result)





if __name__=='__main__':
    test()



