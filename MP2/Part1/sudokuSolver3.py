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


# eliminate v from other_values[k]
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
    # remove word from the the wordbank if it is not used

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
    #  return the domain of the variable i.e. ['H',(0,0)],
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
    if letter != values[coord] and values[coord]!= '': # when b = b, overlap
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


def check_variable2(word, values):
    index = -1
    for i in range(0, 9):
        for j in range(0, 9):
            if values[(i, j)] == '':
                index = [i, j]
                break
        if index != -1:
            break
    print("INDEX: ", index)
    if index == -1:
        return False
    domain = []


    row_list=[]
    col_list=[]

    startC = max(index[1]-len(word)+1,0)
    stopC = min(index[1], 9-len(word))+1

    for i in range(startC, stopC):
        if (values[(index[0], i)] == ''):
            row_list.append((index[0], i))
        elif (values[(index[0], i)] == word[0]):
            row_list = [(index[0], i)]
            break
    startR = max(index[0] - len(word) + 1, 0)
    stopR = min(index[0], 9 - len(word))+1

    for i in range(startR, stopR):
        if (values[(i, index[1])] == ''):
            col_list.append((i,index[1]))
        elif (values[(i,index[1])] == word[0]):
            col_list = [(i,index[1])]
            break

    print("Rowlist: ", row_list)
    print("Collist: ", col_list)
    if row_list:
       for row_co in row_list:
           domain.append(['H', row_co])
    if col_list:
        for col_co in col_list:
            domain.append(['V', col_co])

    print(domain)
    if len(domain) == 0:
        return False
    return domain


def assign_values(word, values, coord, order):

    if order == 'H':
        for i in range(len(word)):
            values[(coord[0], coord[1]+i)] = word[i]
    if order == 'V':
        for i in range(len(word)):
            values[(coord[0]+i, coord[1])] = word[i]

    return values


def d_filter(word, values, domains):
    domain =[]
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


def recursive_dfs(values, wordbank):

    not_used = []
    if all(len(values[s]) == 1 and values[s] != '' for s in squares):
        #sol_set.append(values)
        return values

    word = select_variable(wordbank)
    domains = check_variable(word, values)
    for i in domains:
        coord = i[1]
        order = i[0]
        if check_word(word, values, coord, order):
            new_values = copy.deepcopy(values)
            new_values = assign_values(word, new_values, coord, order)
            print(get_values(new_values))

            new_wordbank = copy.deepcopy(wordbank)
            new_wordbank.remove(word)

            res = recursive_dfs(new_values, new_wordbank)
            if res:
                return res

    global count
    count += 1
    print("Count:", count, "Word: ",word)

    not_used.append([count, word])
    print(not_used)


def get_values(values):
    for i in range(0, 9):
        temp_list = []
        for j in range(0, 9):
            temp_list.append(values[(i, j)])
        print(temp_list)


def select_queue(wordbank):
    temp_wordbank = copy.deepcopy(wordbank)
    candidates = []
    while len(temp_wordbank) > 0:
        word = select_variable(temp_wordbank)
        candidates.append(word)
        temp_wordbank.remove(word)
    #print("The candidates list: \n", candidates)
    return candidates


def traverse(values, wordbank):

    global sol_set
    start_bank = select_queue(wordbank)  # get the word list arranged by length
    start_bank = start_bank[::-1]

    frontier = []
    circle = 0
    word = start_bank[-1]

    start_bank.remove(word)
    domains = check_variable(word,values)

    start_parent = None

    stack = collections.deque([])
    domain = d_filter(word, values, domains)
    print("domain", domain)
    for i in domain:

        frontier.append([word, values, i[0], i[1], start_bank, start_parent])  # frontier is ['operating',values,'H',(0,0),wordbank,parent]
    for j in frontier:
        stack.append(j)
    #stack.append(frontier[1])

    while len(stack) > 0:
        circle += 1
        print(circle)
        node = stack.pop()
        temp_word = node[0]
        temp_values = node[1]
        temp_order = node[2]
        temp_coord = node[3]
        temp_bank = node[4]
        parent = node[5]


        if check_word(temp_word, temp_values, temp_coord, temp_order) == False:
            continue
        if get_empty(temp_values) <= 39 and len(temp_word) > 5:
            continue

        next_values = assign_values(temp_word, temp_values, temp_coord, temp_order)
        # assign the word to the grid and get the updated grid


        if all(len(next_values[s]) == 1 and next_values[s] != '' for s in squares):
            sol_set.append([next_values, ([temp_word, temp_order, temp_coord], parent)])
            return sol_set

        print(get_values(next_values))
        print(temp_word)
        #temp_bank = select_variable2(temp_bank,values)
        for i in temp_bank:

            next_domains = check_variable(i, next_values)
            if next_domains == False:
                continue
            next_domain = d_filter(i, next_values, next_domains)

            if next_domain == False:
                continue

            temp_frontier = []
            next_wordbank = copy.deepcopy(temp_bank)
            next_wordbank.remove(i)

            #if not inf(next_wordbank, next_values):
            #    print("I AM HERE: ",temp_word)
            #    continue

            for j in next_domain:
                next_order = j[0]
                next_coord = j[1]
                temp_frontier.append([i, next_values, next_order, next_coord, next_wordbank, ([temp_word, temp_order, temp_coord],parent)])
            for k in temp_frontier:
                stack.append(k)

    return False

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

explored = set()

def recusive_test(values, wordbank):

    global explored, sol_set
    if all(len(values[s]) == 1 and values[s] != '' for s in squares):
        #sol_set.append(values)
        return values

    word = select_variable(wordbank)
    domains = check_variable(word, values)

    for i in domains:
        coord = i[1]
        order = i[0]
        if check_word(word, values, coord, order):

            new_values = copy.deepcopy(values)
            new_values = assign_values(word, new_values, coord, order)

            explored.add(word)
            print(get_values(values))
            new_wordbank = copy.deepcopy(wordbank)
            new_wordbank.remove(word)
            #nod = getRemainingValues(wordbank,values)
            #if not inference(values,nod):
            #    continue
            res = recusive_test(new_values, new_wordbank)
            if res:
                return res

def get_empty(values):
    count = 0
    for i in range(0,9):
        for j in range(0,9):
            if values[(i,j)] == '':
                count += 1
    return count



def removeInconsistent(values,nodeList):
    pass

def find_blanks(values):
    pass # should find all the blank cells in the grid
def check_blanks(wordbank,list):
    pass # should put the remaining word in the bank into the grid


def traverse_tree(wordbank, values):

    if all(len(values[s]) == 1 and values[s] != '' for s in squares):
        sol_set.append(values)
        return values

    curr_index = find_variable(values)
    domain = find_domain(curr_index, wordbank, values)
    print(domain)
    for d in domain:
        word = d[0]
        order = d[1][0][0]
        coord = d[1][0][1]

        new_values = copy.deepcopy(values)
        new_values = assign_values(word, new_values, coord, order)
        print(get_values(new_values))
        new_wordbank = copy.deepcopy(wordbank)
        new_wordbank.remove(word)
        if not inf(new_wordbank, new_values):
            print("I AM HERE: ", word)
            continue
        res = traverse_tree(new_wordbank, new_values)
        if res:
            return res
    return False


def fill(word, i_values, order, coord):
    values = copy.deepcopy(i_values)
    if order == 'H':
        if coord[1] + len(word)>9:
            return False
        for i in range(len(word)):
            if values[(coord[0],coord[1]+i)] == '':
                values[(coord[0],coord[1]+i)] = word[i]
            elif values[(coord[0],coord[1]+i)] != word[i]:
                return False
        return True
    if order == 'V':
        if coord[0] + len(word)>9:
            return False
        for i in range(len(word)):
            if values[(coord[0]+i,coord[1])] == '':
                values[(coord[0]+i,coord[1])] = word[i]
            elif values[(coord[0]+i,coord[1])] != word[i]:
                return False
        return True
    return False


def traverse_tree2(wordbank, values):
    start_cell = find_variable(values)
    start_domain = find_domain(start_cell,wordbank,values)

    stack = collections.deque([])
    stack.append([start_cell, values, wordbank])


# find a blank cell in the grid, return the variable
def find_variable(values):
    index = -1
    for i in range(0, 9):
        for j in range(0, 9):
            if values[(i, j)] == '':
                index = [i, j]
                break
        if index != -1:
            return index

    return False


# find valid values (i.e. word for the cell), return domains
# input: variable[i,j], wordbank, grid
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


def inference(word,values):
    return removeInconsistent(word, values)


def getRemainingValues(word,values):
    temp_list = []  # the list maintains the remaining variables in the grid
    for i in range(0,9):
        for j in range(0,9):
            if values[(i,j)] == '':
                temp_list.append([i,j])

    for var in temp_list:
        pass


def inf(wordbank,values):
    temp_list = []  # the list maintains the remaining cells in the grid
    for i in range(0, 9):
        for j in range(0, 9):
            if values[(i, j)] == '':
                temp_list.append((i,j))
    word_list = []

    for c in temp_list:
        temp = find_domain(c, wordbank, values)
        word_list.append(temp)
    for i in word_list:
        if len(i) == 0:
            return False
    return True

def count_char_exist(word, order, coord, values):
    count = 0
    y = coord[1]
    x = coord[0]
    origin_string = ""
    if order == 'V':
        origin_string = [values[(y + i,x)] for i in range(len(word))]
    elif order == 'H':
        origin_string = [values[(y,x + i)] for i in range(len(word))]
    for i in range(len(word)):
        if origin_string[i] != word[i] and origin_string[i] != '':
            return 0
        if origin_string[i] == word[i]:
            count += 1
    return count

def organizer():
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

    res = traverse(values, wordbank)
    #f = open('res.txt', 'w')
    #for i in res:
    #    f.write(str(i))
    #    f.write("\n")
    #f.close()

    print(get_values(res[0][0]))
    print("------------------------------------------")
    for i in res[0][1]:
        print(i)




def test():
    global explored,sol_set
    grid = gen_matrix('grid3.txt')
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i][j] != '_':
                grid[i][j] = grid[i][j].lower()
            if grid[i][j] == '_':
                grid[i][j] = ''
    wordbank = get_wordbank('bank3.txt')

    start_bank = select_queue(wordbank)
    values = grid2values(grid)
    print(values)
    start = time.clock()
    #wordbank.remove('ambush')
    test = traverse(values, wordbank)
    #test = traverse_tree(start_bank, values)
    print(test)
    #res = recusive_test(values, wordbank)
    #print(get_values(res[0]))
    #res = explored

    interval = time.clock() - start
    print("The running time is: ", interval,"s")

if __name__ == '__main__':
    test()
