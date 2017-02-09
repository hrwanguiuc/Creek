#!/usr/bin/python
import collections
import heapq

def read_data(name):
    with open(str(name),'r') as f:
        lines = f.readlines()
        maze=[]
        for i in range(len(lines)):
            maze.append(list(lines[i].strip()))

    #print((maze[20]))
    return maze

def bfs(x,y,maze):
    temp=[]
    visited =set() # create a set to store visited points in the maze
    q = collections.deque([(x,y,None)]) # create a queue
    while len(q)>0:
        nodeList = q.popleft() # use popleft() to implement FIFO
        x = nodeList[0]
        y = nodeList[1]
        if (maze[x][y] == '.'): # the goal state
            print (len(visited)) #total expanded nodes in the search
            return (pathToGoalState(nodeList),len(visited))
        if ((x,y) in visited):
            continue
        if (maze[x][y]=='%'):# the situation when the node is not path
            continue
        visited.add((x,y))# indicate the node has been visited
        for i in [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]:  # move up, down, left and right
            q.append((i[0],i[1],nodeList)) # combine the current coordinate and parent as new node and add it to the queue
            #print(nodeList)
    return temp

def pathToGoalState(nodeList):
    path = [] # create a null list
    while (nodeList != None):
        path.append((nodeList[0],nodeList[1]))  #recurvisely add node into the path list.
        nodeList = nodeList[2]
    return path # return a list of (x,y) pairs, which is the path to the goal state

def plotPath(path,maze,name,cost,expanded):
    for i in range(len(maze)):
        for j in range(len(maze[i])):
            if ((i,j) in path):
                maze[i][j] = '.'    # change the character ' ' to '.' in the maze
    with open(str(name),'w') as f: # open a new file
        for i in range(len(maze)):
            for j in range(len(maze[i])):
                f.write(str(maze[i][j])) # write each element in the list into a file.
            f.write('\n')
        f.write('\n')
        f.write('-----------------------------------------------------\n')
        f.write("The total number of dots in the path is: ")
        f.write(str(cost))
        f.write('\n')
        f.write("The number of expanded nodes is: ")
        f.write(str(expanded))
        f.write('\n')


def dfs(x,y,maze):
    temp = []
    visited = set()
    stack = collections.deque([(x,y,None)]) # create a stack in DFS search strategy
    startAdj=[(x,y-1),(x,y+1),(x+1,y),(x-1,y)]

    count =0
    while len(stack) > 0:
        nodeList = stack.pop()  # use pop() to implement LIFO
        x = nodeList[0]
        y = nodeList[1]

        if (maze[x][y] == '.'):  # the goal state
            #print (len(visited)) #total expanded nodes in the search
            return (pathToGoalState(nodeList),len(visited))
        if ((x, y) in visited):
            continue
        if (maze[x][y] == '%'):  # the situation when the node is not path
            continue
        visited.add((x, y))  # indicate the node has been visited
        for i in [ (x-1, y ), (x, y + 1),(x , y-1),(x + 1, y) ]:  # move left, up, down and right
            stack.append((i[0], i[1], nodeList))
            '''combine the current coordinate and parent as new node and add it to the stack'''
            # print(nodeList)
    return temp

def heuristic(start_x,start_y,goal_x,goal_y):

    manhattan = abs(start_x-goal_x) + abs(start_y - goal_y)
    return manhattan

def greedy_best_first(x,y,maze):
    visited = set()
    temp = []
    for i in range(len(maze)):
        for j in range(len(maze[i])):
            if maze[i][j] == '.':
                goal_x = i
                goal_y = j
    priority_queue =[]
    heapq.heappush(priority_queue, (heuristic(x,y,goal_x,goal_y),(x,y,None)))
    '''push the start point with its heuristic value into the prioprity queue'''
    while len(priority_queue)>0:

        nodeList = heapq.heappop(priority_queue)
        x = nodeList[1][0]
        y = nodeList[1][1]
        if (maze[x][y] == '.'):  # the goal state
            print(len(visited))
            return (pathToGoalState(nodeList[1]),len(visited))
        if ((x, y) in visited): # handle the repeated situation
            continue
        if (maze[x][y] == '%'):  # the situation when the node is not path
            continue
        visited.add((x, y))  # indicate the node has been visited
        for i in [ (x - 1, y),(x, y + 1), (x, y - 1),(x + 1, y)]:  # move down, right, up and left
            heapq.heappush(priority_queue,(heuristic(i[0],i[1],goal_x,goal_y),(i[0],i[1],nodeList[1])))
            '''combine the heuristic value of the current node and its parent as
            new node and add it to the priority queue'''
        #print(nodeList)
    return temp

def aStar(x,y,maze):
    visited = set()
    #temp = []
    cost = 0
    bestPath= None
    for i in range(len(maze)):
        for j in range(len(maze[i])):
            if maze[i][j] == '.':
                goal_x = i
                goal_y = j
    priority_queue = []
    heapq.heappush(priority_queue, ((cost + heuristic(x, y, goal_x, goal_y)), cost, (x, y, None)))
    '''push the start point with its heuristic value into the prioprity queue'''
    while len(priority_queue) > 0:
        nodeList = heapq.heappop(priority_queue)
        x = nodeList[2][0]
        y = nodeList[2][1]
        cost = nodeList[1]
        if (maze[x][y] == '.'):  # the goal state

            if bestPath is None or len(pathToGoalState(nodeList[2]))< len(bestPath):
                bestPath = pathToGoalState(nodeList[2])
            #return (pathToGoalState(nodeList[2]),len(visited))
        if ((x, y) in visited):  # handle the repeated situation
            continue
        if (maze[x][y] == '%'):  # the situation when the node is not path
            continue
        visited.add((x, y))  # indicate the node has been visited
        for i in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:  # left,right,down,up
            new_eval = cost + heuristic(i[0], i[1], goal_x, goal_y)
            heapq.heappush(priority_queue, (new_eval, cost+1, (i[0], i[1], nodeList[2])))
            '''combine the evaluation function value(cost+heuristic) of the current node and its parent as
            new node and add it to the priority queue'''
        #print(nodeList)
    temp = bestPath
    return (temp,len(visited))

def main():
    filename = input("Please enter the filename in the search problem: (Hint: mediumMaze.txt, bigMaze.txt, openMaze.txt)\n")
    maze = read_data(filename)
    searchStrategy = input("Please select the searching strategy: (Hint: bfs,dfs,greedy,astar)\n")

    expanded = 0
    for i in range(len(maze)):
        for j in range(len(maze[i])):
            if (maze[i][j] == 'P'):
                x = i
                y = j
    if (searchStrategy == "bfs"):
        result = bfs(x,y,maze)
    if (searchStrategy == "dfs"):
        result = dfs(x,y,maze)
    if (searchStrategy == "greedy"):
        result = greedy_best_first(x,y,maze)
    if (searchStrategy =="astar"):
        result = aStar(x,y,maze)
    #result = bfs(x, y, maze)
    #result = dfs(x, y, maze)
    #result = greedy_best_first(x,y,maze)
    #result = aStar(x,y,maze)
    path = result[0]
    expanded = result[1]

    plotPath(path, maze, "result_"+searchStrategy+"_"+filename,len(path),expanded)
    print("The path to the goal is :")
    print(path)
    print("The total cost of the path is: ", len(path))
    print("The number of expanded Node is :", expanded)
if __name__=='__main__':
    main()