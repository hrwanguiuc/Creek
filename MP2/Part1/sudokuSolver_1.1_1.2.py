import copy
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
#    Initialization of variables            #
#############################################

rows = '012345678'
cols = '012345678'
# constraints for the square
squares = cross(rows, cols)  # squares is a tuple e.g. (1,1)
colConst = [cross(rows, c) for c in cols]
rowConst = [cross(r, cols) for r in rows]
gridConst = [cross(row, col) for row in ('012', '345', '678') for col in ('012', '345', '678')]
constraints = (rowConst + colConst + gridConst)


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
    maxlen = len(temp)

    for i in wordbank:
        if len(i) > maxlen:
            temp = i
            maxlen = len(temp)
    return temp


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
    # return the domain of the variable i.e. ('H',(0,0)),
    #  coord is the coordinate of the first letter of the word



# find all possible values
# index is all possible position for the first letter
def row_check(index, values, word):
    row_list = []
    for i in range(0,9):
        if (word[0] == values[(i,index)]):
            return [(i,index)]
        if (values[i,index] == ''):
            row_list.append((i,index))
    if len(row_list) == 0:
        return False
    return row_list


def col_check(index, values, word):
    col_list = []
    for i in range(0, 9):
        if word[0] == values[(index, i)]:
            return [(index,i)]
        elif values[(index, i)] == '':
            col_list.append((index, i))
    if len(col_list) == 0:
        return False
    return col_list


# check whether a single letter violates constraints or not
def check_constraints(letter,values,coord):
    const_list = []
    if (letter != values[coord] and values[coord]!= ''): # when b = b, overlap
        return False
    for i in peers[(coord)]:
        const_list.append(values[i])
    if letter in const_list:
        return False
    else:
        return True


# try to assign the word to the grid
def check_word(word, values, coord, order):
    if order == 'H':
        if not all(check_constraints(word[i], values, (coord[0],coord[1]+i)) for i in range(len(word))):
            return False
        return True
    if order == 'V':
        if not all(check_constraints(word[i], values, (coord[0]+i, coord[1])) for i in range(len(word))):
            return False

        return True


def assign_values(word, values, coord, order):

    if order == 'H':
        for i in range(len(word)):
            values[(coord[0],coord[1]+i)] = word[i]
    if order == 'V':
        for i in range(len(word)):
            values[(coord[0]+i,coord[1])] = word[i]

    return values


def d_filter(word, values, domains):
    domain =[]
    for i in domains:
        order = i[0]
        coord = i[1]
        if check_word(word,values,coord,order):
            domain.append(i)
    if len(domain) == 0:
        return False
    return domain

sequence = []
count = 0
def recursive_dfs(values, wordbank):
    global sequence,count
    count +=1
    if all(len(values[s]) == 1 and values[s] != '' for s in squares):
        return values
    word = select_variable(wordbank)
    domains = check_variable(word, values)
    for i in domains:
        coord = i[1]
        order = i[0]
        if check_word(word, values, coord, order):
            new_values = copy.deepcopy(values)
            new_values = assign_values(word, new_values, coord, order)
            new_wordbank = copy.deepcopy(wordbank)
            new_wordbank.remove(word)
            if len(sequence)>0:
                for i in sequence:
                    if word == i[2]:
                        sequence.remove(i)
            sequence.append([order,coord,word])
            res = recursive_dfs(new_values, new_wordbank)
            if res:
                return res
    return False


def get_values(values):
    for i in range(0, 9):
        temp_list = []
        for j in range(0, 9):
            temp_list.append(values[(i, j)])
        print(temp_list)


def write_seq():
    global sequence
    f = open('output1.2.txt','w')
    for i in sequence:
        word=i[2]
        order=i[0]
        coord = i[1]
        f.write(str(order)+','+str(coord)+','+str(word)+"\n")
    f.close()




def organizer():
    global sequence,count
    grid = gen_matrix('grid2.txt')
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i][j] != '_':
                grid[i][j] = grid[i][j].lower()
            if grid[i][j] == '_':
                grid[i][j] = ''
    wordbank = get_wordbank('bank2.txt')
    values = grid2values(grid)
    print(values)
    st = time.clock()
    res = recursive_dfs(values,wordbank)
    #print(time.clock()-st)
    print(res)
    file = open('sol_2.txt','w')

    for i in range(0, 9):
        temp_list = []
        for j in range(0, 9):
            temp_list.append(res[(i, j)])
        print(temp_list)
        file.write(str(temp_list))
        file.write('\n')
    file.close()

    print("The sequence is:", sequence)
    write_seq()
    print("The number of expanded nodes is: ", count)
    print("The running time is: ",time.clock()-st,"s")

    #print(check_word('lighten',values,(0,0),'H'))

    '''
    word = select_variable(wordbank)
    print("Word is: ", word)
    d = check_variable(word, values)
    print(d)
    #coord = d[0][1]
    #order = d[0][0]
    #res = check_word('marveling',values,coord,order)
    #print (res)

    res = 0
    for i in d:
        order = i[0]
        coord = i[1]
        if check_word(word, values, coord, order):
            temp_values = copy.deepcopy(values)
            new_values = assign_values(word, temp_values, coord, order)
            new_wordbank = copy.deepcopy(wordbank)
            new_wordbank.remove(word)
            res = new_values
            print("order is: ", order)
            print("coord is: ", coord)
    print("Final result: ",res)
    '''





if __name__ == '__main__':
    organizer()
'''
    res = assign_values('marveling', values, (1,0), 'H')
    for i in range(0, 9):
        temp_list = []
        for j in range(0, 9):
            temp_list.append(res[(i, j)])
        print(temp_list)
    '''
'''
    if all(len(values[s]) == 1 and values[s] != '' for s in squares):
        return values
    word = select_variable(wordbank)  # find a variable
    print("Word is: ", word)
    print(get_values(values))
    domains = check_variable(word, values)  # find the basic domains of a specific variable
    domain = d_filter(word, values, domains)
    for i in domain:
        order = i[0]
        coord = i[1]
        if check_word(word,values,coord,order):
            values = assign_values(word, values, coord, order)
            wordbank.remove(word)
            #return (order,coord,word)
            res = recursive_dfs(values, wordbank)
            if res:
                return res

    return False
'''
