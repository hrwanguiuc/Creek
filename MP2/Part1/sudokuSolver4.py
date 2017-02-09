import copy
import collections
import time

from functools import *



# read the sudoku grid from the file and convert it into a matrix
def gen_matrix(filename):
    with open(str(filename), 'r') as f:
        lines = f.readlines()
        grid = []
        for i in range(len(lines)):
            grid.append(list(lines[i].strip()))
    return grid


def get_wordbank(filename):
    with open(str(filename), 'r') as f:
        lines = f.readlines()
    return [i.strip() for i in lines]


# return a tuple of elements in A and B

def cross(A, B):
    temp = []
    for i in A:
        for j in B:
            temp.append((int(i), int(j)))
    return temp


#############################################
#    Initialization of values               #
#############################################

rows = '012345678'
cols = '012345678'
# constraints for the square
squares = cross(rows, cols)  # squares is a tuple e.g. (1,1)
colConst = [cross(rows, c) for c in cols]
rowConst = [cross(r, cols) for r in rows]
gridConst = [cross(row, col) for row in ('012', '345', '678') for col in ('012', '345', '678')]
constraints = (rowConst + colConst + gridConst)

sol_set = []


def gen_unitConst():
    temp = {}
    for k in squares:
        vList = []
        for v in constraints:
            if k in v:
                vList.append(v)
        temp[k] = vList
    return temp


# get peers for each unit
def gen_peers():
    temp = {}
    for k in squares:
        tempSet = set()
        tempList = reduce(lambda x, y: x + y, units[k], [])
        for i in tempList:
            tempSet.add(i)
        tempSet.remove(k)
        temp[k] = tempSet
    return temp


units = gen_unitConst()
peers = gen_peers()


########################################################
## Function to parse the grid and manipulations       ##
########################################################


# parse the input grid and output all the values according to the initial state of the grid
def parse_grid(grid):
    values = {}
    cur_grid_dict = current_grid(grid)

    return cur_grid_dict  # values is a dictionary


def current_grid(grid):
    templist = []
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            # tempList is used to get the value of each unit in the grid,from (0,0)->(0,8)->(1,0)->(1,8)->...->(8,8)
            templist.append(grid[i][j])

    tempDict = dict(zip(squares, templist))
    return tempDict


def assign(values, k, v):
    other_values = values[k].replace(v, '')  # to create other values which does not include v
    for i in other_values:
        if not eliminate(values, k, i):
            return False
    return values


# eliminate v from ohter_values[k]
# k stands for key, v stands for value


def eliminate(values, k, v):
    if v not in values[k]:
        return values
    values[k] = values[k].replace(v, '')
    if len(values[k]) == 0:  # contradiction
        return False
    elif len(values[k]) == 1:  # if the possible value for values[k] is only 1, then eliminate this value in its peers
        new_v = values[k]
        if not all(eliminate(values, new_k, new_v) for new_k in peers[k]):
            return False
    return values


def grid2values(grid):
    return parse_grid(grid)


####################################
## Function for assignment        ##
####################################

# find the max length of the word as the next variable
def select_variable(wordbank):
    temp = wordbank[0]
    maxlen = -1
    # remove word from the the wordbank if it is not used

    for i in wordbank:
        if len(i) > maxlen:
            temp = i
            maxlen = len(temp)
    return temp

def select_variable2(wordbank, values):
    temp =''
    maxlen = -1
    empty_cells = 0

    searching_length = -1

    for i in range(0,9):
        for j in range(0,9):
            if values[(i,j)] == '':
                empty_cells +=1
    if empty_cells <= 35:
        searching_length = 5
    elif empty_cells <= 20:
        searching_length = 9

    else:
        searching_length = 9
    for word in wordbank:
        if len(word) > maxlen and len(word) <= searching_length :
            temp = word
            maxlen = len(word)

    if maxlen == -1:
        temp = wordbank[0]
    return temp


def domain_list(word, values):
    index = 0
    limit = 9 - len(word)
    domain = []

    while (index <= limit):

        row_list = row_check(index, values, word)
        col_list = col_check(index, values, word)

        if row_list:
            for row_co in row_list:
                if not check_word(word, values, row_co, 'H'):
                    continue
                domain.append(['H', row_co])
        if col_list:
            for col_co in col_list:
                if not check_word(word, values, col_co, 'V'):
                    continue
                domain.append(['V', col_co])
        index += 1
    #domain.sort(key=lambda d: self.count_char_exist(d, word, values), reverse=True)
    domain.append(['N', (0, 0)])
    return domain


# get domains for a variable
def check_variable(word, values):
    index = 0
    limit = 9 - len(word)
    domain = []

    while (index <= limit):

        row_list = row_check(index, values, word)
        col_list = col_check(index, values, word)

        if row_list:
            for row_co in row_list:
                domain.append(['H', row_co])

        if col_list:
            for col_co in col_list:
                domain.append(['V', col_co])

        index += 1
    if len(domain) == 0:
        return False
    return domain
    #  return the domain of the variable i.e. ['H',(0,0)],
    #  coord is the coordinate of the first letter of the word


# find all possible values
# index is all possible position for the first letter
def row_check(index, values, word):
    row_list = []
    for i in range(0, 9):
        if (word[0] == values[(i, index)]):
            return [(i, index)]
        if (values[i, index] == ''):
            row_list.append((i, index))
    if len(row_list) == 0:
        return False
    return row_list


def col_check(index, values, word):
    col_list = []
    for i in range(0, 9):
        if word[0] == values[(index, i)]:
            return [(index, i)]
        elif values[(index, i)] == '':
            col_list.append((index, i))
    if len(col_list) == 0:
        return False
    return col_list


# check whether a single letter violates constraints or not
def check_constraints(letter, values, coord):
    const_list = []
    if letter != values[coord] and values[coord] != '':  # when b = b, overlap
        return False
    for i in peers[(coord)]:
        const_list.append(values[i])
    if letter in const_list:
        return False
    else:
        return True


# try to assign the word to the grid to see whether it satisfies the constraints
def check_word(word, values, coord, order):
    if order == 'H':
        if not all(check_constraints(word[i], values, (coord[0], coord[1] + i)) for i in range(len(word))):
            return False
        return True
    if order == 'V':
        if not all(check_constraints(word[i], values, (coord[0] + i, coord[1])) for i in range(len(word))):
            return False
        return True

# update the grid
def assign_values(word, values, coord, order):
    if order == 'H':
        for i in range(len(word)):
            values[(coord[0], coord[1] + i)] = word[i]
    if order == 'V':
        for i in range(len(word)):
            values[(coord[0] + i, coord[1])] = word[i]
    if order == 'N':
        pass
    return values

# filter the domains
def d_filter(word, values, domains):
    domain = []
    for i in domains:
        order = i[0]
        coord = i[1]
        if check_word(word, values, coord, order) == False:
            continue
        domain.append(i)
    if len(domain) == 0:
        return False
    return domain


count = 0

seq = []
used = set()

def recursive_dfs(values, wordbank):

    global seq, sol_set, used
    if all(len(values[s]) == 1 and values[s] != '' for s in squares):
        sol_set.append(values)
        return values
    if len(wordbank) == 0:
        return False
    #word = selectWord(wordbank, values)
    word = select_variable2(wordbank,values)
    print("SELECTED WORD:",word)
    domains = domain_list(word,values)
    #domains = check_variable(word, values)
    #domain = d_filter(word,values,domains)
    '''
    if not domains:
        next_wordbank = copy.deepcopy(wordbank)
        next_wordbank.remove(word)
        next_values = copy.deepcopy(values)
        return recursive_dfs(next_values, next_wordbank)
    '''

    for i in domains:
        coord = i[1]
        order = i[0]
        if check_word(word, values, coord, order):
            new_values = copy.deepcopy(values)
            new_values = assign_values(word, new_values, coord, order)
            print(get_values(new_values))
            new_wordbank = copy.deepcopy(wordbank)
            new_wordbank.remove(word)
            used.add(word)
            res = recursive_dfs(new_values, new_wordbank)
            if res:
                return res
    return False
    global count
    count += 1
    print("Count: ", count, " word: ", word)
    seq.append(word)
    '''
    new_wordbank = copy.deepcopy(wordbank)
    new_wordbank.remove(word)
    new_values = copy.deepcopy(values)
    return recursive_dfs(new_values, new_wordbank)
    '''



def find_domain(index, wordbank, values):

    wordbank_list = []  # [operating , [['H', (0,1)]]]

    for word in wordbank:
        domain = []
        row_list = []
        col_list = []
        startC = max(index[1] - len(word) + 1, 0)
        stopC = min(index[1], 9 - len(word)) + 1

        for i in range(startC, stopC):
            if (values[(index[0], i)] == ''):
                row_list.append((index[0], i))
            elif (values[(index[0], i)] == word[0]):
                row_list = [[index[0], i]]      # return a coordinate
                break
        startR = max(index[0] - len(word) + 1, 0)
        stopR = min(index[0], 9 - len(word)) + 1

        for i in range(startR, stopR):
            if (values[(i, index[1])] == ''):
                col_list.append((i, index[1]))
            elif (values[(i, index[1])] == word[0]):
                col_list = [(i, index[1])]
                break

        #print("Rowlist: ", row_list)
        #print("Collist: ", col_list)
        if row_list:
            for row_co in row_list:
                domain.append(['H', row_co])
        if col_list:
            for col_co in col_list:
                domain.append(['V', col_co])

        #print(domain)
        if len(domain) == 0:
            continue
        domain_filtered = d_filter(word, values, domain)    # this is final domain for the word
        if not domain_filtered:
            continue

        wordbank_list.append([word, domain_filtered])

    return wordbank_list




def get_values(values):
    for i in range(0, 9):
        temp_list = []
        for j in range(0, 9):
            temp_list.append(values[(i, j)])
        print(temp_list)



def selectWord(wordbank, values):
    dictionary = {}
    for ch in "abcdefghijklmnoprstuvwxyz":
        dictionary[ch] = 0
    for i in range(0,9):
        for j in range(0,9):
            if values[(i,j)] != '':
                dictionary[values[(i,j)]] += 1
    # maxlen = 0
    maxweight = 0
    returnword = ''
    for word in wordbank:
        weight = 0
        for ch in word:
            weight += dictionary[ch]
        if len(word) > len(returnword) or (len(word) == len(returnword) and weight > maxweight):
            returnword = word
            maxweight = weight
    return returnword

def moniter(values, wordbank):
    global seq
    while len(wordbank) > 0:
        recursive_dfs(values, wordbank)



def test():

    global explored, sol_set
    grid = gen_matrix('grid3.txt')
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i][j] != '_':
                grid[i][j] = grid[i][j].lower()
            if grid[i][j] == '_':
                grid[i][j] = ''
    wordbank = get_wordbank('bank3.txt')
    values = grid2values(grid)
    print(values)
    start = time.clock()

    res = recursive_dfs(values, wordbank)
    # res = explored
    print(get_values(res))
    print("seq: ", seq)
    print("used: ", len(used))
    interval = time.clock() - start
    print("The running time is: ", interval, "s")
    print(len(sol_set))


if __name__ == '__main__':
    test()
'''
    def count_char_exist(self, domain, word, values):
        count = 0
        coord = domain[1]
        order = domain[0]
        y = coord[0]
        x = coord[1]
        origin_string = ""
        if order == 'V':
            origin_string = [values[(y + i,x)] for i in range(len(word))]
        elif order == 'H':
            origin_string = [values[(y, x + i)] for i in range(len(word))]
        for i in range(len(word)):
            if origin_string[i] != word[i] and origin_string[i] != '':
                return 0
            if origin_string[i] == word[i]:
                count += 1
        return count
'''