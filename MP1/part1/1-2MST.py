#!/usr/bin/python

import heapq
import string
import collections


def read_data(name):
    with open(str(name),'r') as f:
        lines = f.readlines()
        maze=[]
        for i in range(len(lines)):
            maze.append(list(lines[i].strip()))

    return maze

def findAllDots(maze):
    #pass
    dotsList = []
    for i in range(len(maze)):
        for j in range(len(maze[i])):
            if (maze[i][j]=='.'):
                dotsList.append((i,j))
    return dotsList


def findTheNearestDot(start,dotsList,filename):
    maze = read_data(filename)
    actualExpanded =0
    x = start[0]
    y = start[1]
    if len(dotsList) == 0:
        return 0

    priority_queue = []

    for node in dotsList:
        #manDist = manhatthanDist(x,y,node[0],node[1])
        #heapq.heappush(priority_queue,(manDist,((x,y),node)))
        actualDist = actualCost(x,y,node[0],node[1],maze)[0]
        actualExpanded += actualCost(x,y,node[0],node[1],maze)[1]
        heapq.heappush(priority_queue, (actualDist, ((x, y), node)))

    nodeList = heapq.heappop(priority_queue)
    if(len(priority_queue)>2):
        nodeList1 = heapq.heappop(priority_queue)

        if (nodeList[0]==nodeList1[0]):# handle tie-breaking, if the distance to two dots are same

            thirdNode = heapq.heappop(priority_queue)
            fourthNode = heapq.heappop(priority_queue)
            cost1 = manhatthanDist(nodeList[1][1][0],nodeList[1][1][1],thirdNode[1][1][0],thirdNode[1][1][1])+ \
                    manhatthanDist(nodeList[1][1][0], nodeList[1][1][1], fourthNode[1][1][0], fourthNode[1][1][1])
            cost2 = manhatthanDist(nodeList1[1][1][0], nodeList1[1][1][1], thirdNode[1][1][0], thirdNode[1][1][1])+\
                    manhatthanDist(nodeList1[1][1][0], nodeList1[1][1][1], fourthNode[1][1][0], fourthNode[1][1][1])
            if (cost1>cost2):
                nodeList = nodeList1

    #print (actualExpanded)
    return (nodeList[1][1],actualExpanded)

def mst(start,dotsList,maze):
    actualExpanded = 0
    x = start[0]
    y = start[1]
    if len(dotsList) == 0:
        return 0

    priority_queue = []
    for node in dotsList:
        total = 0
        manDist = manhatthanDist(x,y,node[0],node[1])
        total = total+ manDist + actualCost(x,y,node[0],node[1],maze)[0]
        heapq.heappush(priority_queue,(total,((x,y),node)))
        actualExpanded += actualCost(x,y,node[0],node[1],maze)[1]
    nodePair = heapq.heappop(priority_queue)
    return (nodePair[1][1],actualExpanded)


# use BFS to find the nearest dot according to the current position
def actualCost(curr_x,curr_y,goal_x,goal_y,maze):
    count =0
    temp=[]
    visited =set() # create a set to store visited points in the maze
    q = collections.deque([((curr_x,curr_y),None)]) # create a queue
    while len(q)>0:
        count +=1
        nodeList = q.popleft() # use popleft() to implement FIFO
        curr_x = nodeList[0][0]
        curr_y = nodeList[0][1]
        if ((curr_x,curr_y)==(goal_x,goal_y)): # the goal state
            #print (len(visited)) #total expanded nodes in the search
            return (len(pathToGoalState(nodeList)),len(visited))
        if ((curr_x,curr_y) in visited):
            continue
        if (maze[curr_x][curr_y]=='%'):# the situation when the node is not path
            continue
        visited.add((curr_x,curr_y))# indicate the node has been visited
        for i in [(curr_x-1, curr_y), (curr_x+1, curr_y), (curr_x, curr_y-1), (curr_x, curr_y+1)]:  # move up, down, left and right
            q.append(((i[0],i[1]),nodeList)) # combine the current coordinate and parent as new node and add it to the queue
            #print(nodeList)
    return temp




def manhatthanDist(start_x,start_y,goal_x,goal_y):
    manhattan = abs(start_x - goal_x) + abs(start_y - goal_y)
    return manhattan

def aStar(start,goal,maze):
        visited = set()
        temp = []
        cost = 0
        start_x = start[0]
        start_y = start[1]
        goal_x = goal[0]
        goal_y = goal[1]
        count = 0
        priority_queue = []
        bestPath = None

        heapq.heappush(priority_queue, ((cost + manhatthanDist(start_x, start_y, goal_x, goal_y)),cost,start,(start,None)))
        '''push the start point with its heuristic value into the priority queue'''
        while len(priority_queue) > 0:
            count += 1
            nodeList = heapq.heappop(priority_queue)
            x = nodeList[2][0]
            y = nodeList[2][1]
            cost = nodeList[1]
            parent = nodeList[3]
            if ((x,y)==(goal_x,goal_y)): # find the goal
                if bestPath is None or len(pathToGoalState(parent))< len(bestPath):
                    bestPath = pathToGoalState(parent)
                    tempParent = parent
                #return (parent,len(visited),cost)
            if ((x, y) in visited):  # handle the repeated situation
                continue
            if (maze[x][y] == '%'):  # the situation when the node is not path
                continue
            visited.add((x, y)) # indicate the node has been visited
            for i in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:  # move up, down, left and right
                new_eval = cost + manhatthanDist(i[0], i[1], goal_x, goal_y)
                heapq.heappush(priority_queue, (new_eval, cost+1, (i[0],i[1]), ((i[0],i[1]),parent)))
                '''combine the evaluation function value(cost+heuristic) of the current node and its parent as
                new node and add it to the priority queue'''
        temp = tempParent
        return (tempParent,len(visited),len(bestPath))


#################################################
def getMST(node,dotsList):
    nodeList =[]
    nodeList.append(node)
    nodeList = nodeList + dotsList
    totalCost= 0
    tempQueue = []
    added = set()
    for i in nodeList:
        for j in nodeList:
            if (i!=j):
                cost = manhatthanDist(i[0],i[1],j[0],j[1]) # cost of manhattan distance between two nodes in the list
                heapq.heappush(tempQueue,(cost,(i,j))) # put the (cost,(node1,node2)) into the priority queue
    print (len(tempQueue))
    while (len(nodeList)!=len(added)):
        temp = heapq.heappop(tempQueue) #pop the smallest element
        firstNode = temp[1][0]
        secondNode = temp[1][1]
        tempCost = temp[0]
        if (firstNode in added) and (secondNode in added):
            continue
        totalCost += tempCost
        added.add(firstNode)
        added.add(secondNode)
        print (temp)
    return totalCost

def Astar_mst(start,dotsList,maze):
    visited = set()
    temp = []
    cost = 0

    count = 0
    priority_queue = []
    bestPath = None

    heapq.heappush(priority_queue,
                   ((cost + getMST(start,dotsList)), cost, start, (start, None)))
    '''push the start point with its heuristic value into the priority queue'''
    while len(priority_queue) > 0:
        count += 1
        nodeList = heapq.heappop(priority_queue)
        x = nodeList[2][0]
        y = nodeList[2][1]
        cost = nodeList[1]
        parent = nodeList[3]
        if ((x, y) in visited):  # handle the repeated situation
            continue
        if (maze[x][y] == '%'):  # the situation when the node is not path
            continue
        if (maze[x][y]=='.'):  # find the dot
            dotsList.remove((x,y))
        #if bestPath is None or len(pathToGoalState(parent)) < len(bestPath):
        #    bestPath = pathToGoalState(parent)
            tempParent = parent
            # return (parent,len(visited),cost)
        visited.add((x, y))  # indicate the node has been visited
        for i in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:  # move up, down, left and right
            new_eval = cost + getMST((i[0],i[1]),dotsList)
            heapq.heappush(priority_queue, (new_eval, cost + 1, (i[0], i[1]), ((i[0], i[1]), parent)))
            '''combine the evaluation function value(cost+heuristic) of the current node and its parent as
            new node and add it to the priority queue'''
    temp = tempParent
    return (temp, len(visited), len(temp))


def MSTtest():
    maze = read_data('tinySearch.txt')
    dotsList = findAllDots(maze)
    for i in range(len(maze)):
        for j in range(len(maze[i])):
            if (maze[i][j] == 'P'):
                x = i
                y = j
    start = (x,y)
    result = Astar_mst(start,dotsList,maze)
    path = pathToGoalState(result[0])
    print (path)
    print (result[1])
    print(result[2])




###################################################################
def pathToGoalState(nodeList):
    path = [] # create a null list
    while (nodeList != None):
        path.append(nodeList[0])  #recurvisely add node into the path list.
        nodeList = nodeList[1]
    return path # return a list of (x,y) pairs, which is the path to the goal state

def plotPath(path,maze,name,cost,expanded):

    orderList= list(range(10)) + list(string.ascii_lowercase) +list(string.ascii_uppercase)
    count = 0
    dotDict={}
    dotsList = findAllDots(maze)
    tempList = dotsList
    for i in path:
        if i in tempList:
            dotDict[i] = orderList[count]
            count+=1
            tempList.remove(i)

    dotsList = findAllDots(maze)
    for i in range(len(maze)):
        for j in range(len(maze[i])):
            if ((i,j) in dotsList):
                maze[i][j] = dotDict[(i,j)]

    with open(str(name),'w') as f: # open a new file
        for i in range(len(maze)):
            for j in range(len(maze[i])):
                f.write(str(maze[i][j])) # write each element in the list into a file.
            f.write('\n')

        f.write('-----------------------------------------------------\n')
        f.write("The total cost of the path is: ")
        f.write(str(cost))
        f.write('\n')
        f.write("The number of expanded nodes is: ")
        f.write(str(expanded))
        f.write('\n')


def main():
    filename = input("Please enter the filename in the search problem: (hint: tinySearch.txt, smallSearch.txt, mediumSearch.txt)\n")
    maze = read_data(filename)
    dotsList = findAllDots(maze)
    print (dotsList)
    print(len(dotsList))
    path = []
    expandedNode_astar = 0
    expandedNode_bfs = 0
    totalCost = 0
    for i in range(len(maze)):
        for j in range(len(maze[i])):
            if (maze[i][j] == 'P'):
                x = i
                y = j
    start = (x,y)

    while len(dotsList)>0:
        goal = findTheNearestDot(start, dotsList,filename)[0] # find the nearest dot for the current start point
        #goal = mst(start,dotsList,maze)[0]
        #expandedNode_bfs += mst(start,dotsList,maze)[1]
        expandedNode_bfs += findTheNearestDot(start, dotsList,filename)[1]
        result = aStar(start,goal,maze)  # repeat the A* search algorithm to find the optimal path to the nearest dot
        tempPath = result[0]
        expandedNode_astar += result[1]
        tempCost = result[2]
        totalCost = totalCost + tempCost
        tempPath = pathToGoalState(tempPath) # get the (x,y) pairs in a list
        tempPath.remove(tempPath[-1])  # remove the last one, which is the start node
        tempPath = tempPath[::-1] # reverse the list to get the actual path
        dotsList.remove(goal)
        for i in tempPath:
            if i in dotsList:
                dotsList.remove(i)
            path.append(i)
        start = goal
    totalExpanded = expandedNode_astar + expandedNode_bfs
    print ("The path to the goal is :")
    print (path)
    print ("The total cost of the path is: ", len(path))
    print ("The number of expanded Node is :",totalExpanded)

    plotPath(path,maze,"result_tinySearch.txt",len(path),totalExpanded)
if __name__=='__main__':
    #main()
    MSTtest()