
import numpy as np


class state():
    def __init__(self, board=None):

        self.board = board
        self.lastmove = (-1,-1)
        self.black_worker_list =[]
        self.white_worker_list =[]
        self.black_num = 0
        self.white_num = 0
        self.row = 0
        self.col = 0
        # init of black_num and white_num
        # init of black_worker_list and w_w_l

        if not self.board:
            for i in range(8):
                for j in range(8):
                    if self.board[i][j] == 1:
                        self.black_num +=1
                        self.black_worker_list.append((i,j))
                    elif self.board[i][j] == 2:
                        self.white_num += 1
                        self.white_worker_list.append((i,j))


    def update_state(self, movement):
        black_worker_list = self.black_worker_list
        white_worker_list = self.white_worker_list
        curr = movement.curr
        dir = movement.dir
        turn = movement.turn


        if turn == 'b':

           if curr in black_worker_list:
               next_coord = move(turn, curr,dir)
               black_worker_list.append(next_coord)
               black_worker_list.remove(curr)
        if turn == 'w':

            if curr in white_worker_list:
                next_coord = move(turn, curr ,dir)
                white_worker_list.append(next_coord)
                white_worker_list.remove(curr)

        new_state = State()







class movement:
    def __int__(self, turn, board, curr, dir):
        self.turn = turn
        self.curr = curr
        self.next = dir




class minimaxAgent():
    def __init__(self):
        self.avgNodes = []
        self.avgTime = []
        self.score = 0
        self.nodes = 0
        self.numMoves = 0

    def cutoff(self, state, depth):
        if depth>= 3:
            return True
        elif np.count_nonzero(state.curState) == 64:
            return True
        else:
            return False

    def if_capture(self, state):
        (last_x,last_y) = state.lastmove
        whiteTurn = False
        if state.curState[last_x][last_y] == 1:
            whiteTurn = True
        total = 0
        if whiteTurn:
            pass
        ##

def move(turn, curr_coord, dir):
    result = (-1,-1)
    if turn == 'b':
        if dir == 'l':
            result = (curr_coord[0]+1,curr_coord[1]-1)
            return result
        if dir =='m':
            result = (curr_coord[0]+1,curr_coord[1])
            return result
        if dir == 'r':
            result = (curr_coord[0]+1, curr_coord[1]+1)
            return result

    elif turn == 'w':
        if dir == 'l':
            result = (curr_coord[0]-1,curr_coord[1]-1)
            return result
        if dir =='m':
            result = (curr_coord[0]-1,curr_coord[1])
            return result
        if dir == 'r':
            result = (curr_coord[0]-1, curr_coord[1]+1)
            return result

    return result





