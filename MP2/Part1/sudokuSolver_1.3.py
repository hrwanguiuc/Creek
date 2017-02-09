
import copy
from functools import *

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
#-----------------------------------------------------------------------------------


class SudokuSolver:
    def __init__(self,values,wordbank):
        self.values =values
        self.wordbank =wordbank
        self.path = []
        self.node_num = 0
        self.existed = set()
        self.sol_list = []

    def select_variable(self, wordbank):
        temp = wordbank[0]
        maxlen = -1
        # remove word from the the wordbank if it is not used

        for i in wordbank:
            if len(i) > maxlen:
                temp = i
                maxlen = len(temp)
        return temp

    def selectWord(self, wordbank, values):
        word_dict = {}
        for char in "abcdefghijklmnoprstuvwxyz":
            word_dict[char] = 0
        for i in range(0, 9):
            for j in range(0, 9):
                if values[(i, j)] != '':
                    word_dict[values[(i, j)]] += 1

        max_weight = 0
        return_word = ''
        for word in wordbank:
            weight = 0
            for char in word:
                weight += word_dict[char]
            if len(word) > len(return_word) or (len(word) == len(return_word) and weight > max_weight):
                # handle tie-breaking
                return_word = word
                max_weight = weight

        return return_word

    def domain_list(self, word, values):
        index = 0
        limit = 9 - len(word)
        domain = []

        while (index <= limit):

            row_list = self.row_check(index, values, word)
            col_list = self.col_check(index, values, word)

            if col_list:
                for col_co in col_list:
                    if not self.check_word(word, values, col_co, 'V'):
                        continue
                    domain.append(['V', col_co])

            if row_list:
                for row_co in row_list:
                    if not self.check_word(word, values, row_co, 'H'):
                        continue
                    domain.append(['H', row_co])

            index += 1
        domain.append(['N', (0, 0)])
        return domain
        #  return the domain of the variable i.e. ['H',(0,0)],
        #  coord is the coordinate of the first letter of the word

    # find all possible values
    # index is all possible position for the first letter
    def row_check(self, index, values, word):
        row_list = []
        for i in range(0, 9):
            if (word[0] == values[(i, index)]):
                return [(i, index)]
            if (values[i, index] == ''):
                row_list.append((i, index))
        if len(row_list) == 0:
            return False
        return row_list

    def col_check(self, index, values, word):
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
    def check_constraints(self, letter, values, coord):
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
    def check_word(self, word, values, coord, order):
        if order == 'H':
            if not all(self.check_constraints(word[i], values, (coord[0], coord[1] + i)) for i in range(len(word))):
                return False
            return True
        if order == 'V':
            if not all(self.check_constraints(word[i], values, (coord[0] + i, coord[1])) for i in range(len(word))):
                return False
            return True


    # update the grid
    def assign_values(self, word, values, coord, order):
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
    def d_filter(self, word, values, domains):
        domain = []
        for i in domains:
            order = i[0]
            coord = i[1]
            if self.check_word(word, values, coord, order) == False:
                continue
            domain.append(i)
        if len(domain) == 0:
            return False
        return domain

    def recursive_dfs(self, values, wordbank):

        if all(len(values[s]) == 1 and values[s] != '' for s in squares):
            #self.sol_list.append(values)
            return values
        if len(wordbank) == 0:
            return False
        #word = self.select_variable(wordbank)
        word = self.selectWord(wordbank, values)
        domains = self.domain_list(word, values)
        print("domains: ", domains)

        for i in domains:
            self.node_num += 1
            coord = i[1]
            order = i[0]
            #
            if self.already_existed(word, values):
                #wordbank.remove(word)
                self.existed.add(word)
                continue


            new_values = copy.deepcopy(values)
            new_values = self.assign_values(word, new_values, coord, order)
            self.path.append([word,order,coord])

            print(self.get_values(new_values))
            new_wordbank = copy.deepcopy(wordbank)
            new_wordbank.remove(word)
            if self.inf(new_wordbank,new_values): # add inference
                res = self.recursive_dfs(new_values, new_wordbank)
                if res:
                    return res
            self.path.remove([word,order,coord])

        return False

    def get_values(self, values):
        for i in range(0, 9):
            temp_list = []
            for j in range(0, 9):
                temp_list.append(values[(i, j)])
            print(temp_list)

    def inf(self,wordbank, values):
        temp_list = []  # the list maintains the remaining cells in the grid
        for i in range(0, 9):
            for j in range(0, 9):
                if values[(i, j)] == '':
                    temp_list.append((i, j))
        word_list = []

        for c in temp_list:
            temp = self.find_domain(c, wordbank, values)
            word_list.append(temp)
        for i in word_list:
            if len(i) == 0:
                return False
        return True

    # check whether the word already appears in the grid
    def already_existed(self, word, values):
        str_list = []
        for r in range(9):
            row_str = ''
            for c in range(9):
                if values[(r,c)]!= '':
                    row_str = row_str+values[(r,c)]
            str_list.append(row_str)
        for c in range(9):
            col_str = ''
            for r in range(9):
                if values[(r,c)] != "":
                    col_str = col_str + values[(r,c)]
            str_list.append(col_str)

        for str in str_list:
            if word in str:
                return True

        return False


    # find valid values (i.e. word for the cell), return domains
    # input: variable[i,j], wordbank, grid
    def find_domain(self, index, wordbank, values):

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
                    row_list = [[index[0], i]]  # return a coordinate
                    break
            startR = max(index[0] - len(word) + 1, 0)
            stopR = min(index[0], 9 - len(word)) + 1

            for i in range(startR, stopR):
                if (values[(i, index[1])] == ''):
                    col_list.append((i, index[1]))
                elif (values[(i, index[1])] == word[0]):
                    col_list = [(i, index[1])]
                    break

            if row_list:
                for row_co in row_list:
                    if not self.check_word(word, values, row_co, 'H'):
                        continue
                    domain.append(['H', row_co])
            if col_list:
                for col_co in col_list:
                    if not self.check_word(word, values, col_co, 'V'):
                        continue
                    domain.append(['V', col_co])
            # print(domain)
            if len(domain) == 0:
                continue
            wordbank_list.append([word, domain])

        return wordbank_list



    def solve(self):
        res = self.recursive_dfs(self.values,self.wordbank)
        print(self.get_values(res))

        for i in self.path:
            if i[0] in self.existed:
                self.path.remove(i)
        print("Sequence is: ", self.path)
        print("Expanded node is: ", self.node_num)
        f = open('output1.3.txt' , 'w')
        for i in self.path:
            if i[1] == 'N':
                continue
            f.write(i[1]+','+str(i[2])+','+i[0]+'\n')
        f.close()


def main():
    grid = gen_matrix('grid3.txt')
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i][j] != '_':
                grid[i][j] = grid[i][j].lower()
            if grid[i][j] == '_':
                grid[i][j] = ''
    wordbank = get_wordbank('bank3.txt')
    values = grid2values(grid)

    sudoku = SudokuSolver(values,wordbank)
    sudoku.solve()
    #print(sudoku.already_existed('wit',values))
if __name__ == '__main__':
    main()