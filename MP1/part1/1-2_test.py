import heapq
import collections
import string

def read_data(name):
    with open(str(name),'r') as f:
        lines = f.readlines()
        maze=[]
        for i in range(len(lines)):
            maze.append(list(lines[i].strip()))

    #print((maze[20]))
    return maze

def findAllDots(maze):
    #pass
    dotsList = []
    for i in range(len(maze)):
        for j in range(len(maze[i])):
            if (maze[i][j]=='.'):
                dotsList.append((i,j))
    return dotsList

def findAllWall(maze):
    wallList = []
    for i in range(len(maze)):
        for j in range(len(maze[i])):
            if (maze[i][j] == '%'):
                wallList.append((i, j))
    return wallList

def heuristic(x,y,dotsList,explored):
    #pass
    #dotsList = findAllDots(maze) # get the list of the coordinates of dots
    edgesDict = {} # use a dictionary to do the query in the next steps

    if len(dotsList) == 0:
        return 0

    priority_queue = []
    '''
    dotsList.append((x,y))
    for node in dotsList:
        for otherNode in dotsList:
            if (node == otherNode):
                continue
            manDist = manhatthanDist(node,otherNode) # calculate the manhattan distance between two dots
            edgesDict[(node,otherNode)] = manDist # add the key-value pair into the dictionary
            heapq.heappush(priority_queue, (manDist,(node,otherNode))) # push the tuple into the prioprity queue
    '''
    '''sum the cost of each edge in the minimum spanning tree'''
    sum = 0

    for node in dotsList:
        manDist = manhatthanDist((x,y),node)
        heapq.heappush(priority_queue,(manDist,((x,y),node)))
    '''
    while len(priority_queue)>0:
        nodeList = heapq.heappop(priority_queue) # get the pair of nodes -> (manDist, (x1,y1),(x2,y2))
        firstNode = nodeList[1][0]
        secondNode = nodeList[1][1]
        nodesPair = nodeList[1]
        if (firstNode not in explored) or (secondNode not in explored):
            sum = sum + edgesDict[nodesPair] # get the sum of all possible manDist between two dots
            #print("This is sum: %d",sum)
            #print("This is edgeDict: %s",edgesDict[nodesPair])
        if (firstNode not in explored):
            explored.add(firstNode)
        if (secondNode not in explored):
            explored.add(secondNode)

    dotsList.remove((x,y))
    return sum
    '''

    nodeList = heapq.heappop(priority_queue)

    return nodeList[0]



def closestDot(x,y,dotsList,first=True):
    edgesDict = {}
    # explored = set()
    priority_queue = []
    temp = ()
    if first:
        dotsList.append((x,y))
    for node in dotsList:
        for otherNode in dotsList:
            if (node == otherNode):
                continue
            manDist = manhatthanDist(node, otherNode)  # calculate the manhattan distance between two dots
            edgesDict[(node, otherNode)] = manDist  # add the key-value pair into the dictionary
            heapq.heappush(priority_queue, (manDist, (node, otherNode)))  # push the tuple into the prioprity queue
    if first:
        dotsList.remove((x,y))
    while len(priority_queue)>0:
        nodeList = heapq.heappop(priority_queue)  # get the pair of nodes -> (manDist, (x1,y1),(x2,y2))
        firstNode = nodeList[1][0]
        secondNode = nodeList[1][1]
        if (firstNode == (x,y)):
            return secondNode
        if (secondNode == (x,y)):
            return firstNode
    return temp

def aStarMultiDots(x,y,maze):
    count = 0
    explored = set()
    visited = []
    temp = []
    cost = 0
    dotsList = findAllDots(maze)
    wallList = findAllWall(maze)
    priority_queue = []
    #nextDot = closestDot(x,y,dotsList)
    evalValue = cost + heuristic(x,y,dotsList,explored) #+ manhatthanDist((x,y),(nextDot[0],nextDot[1]))

    heapq.heappush(priority_queue, (evalValue, cost,(x,y),((x,y,None))))
    '''push the start point with its heuristic value into the prioprity queue'''

    while len(priority_queue) > 0:
        state = heapq.heappop(priority_queue)
        x = state[2][0]
        y = state[2][1]
        cost = state[1]
        parent = state[3]
        sum = 0
        if len(explored) > 0:
            for node in explored:
                sum = sum + node[0]

        if len(dotsList) == 0:
            return infoToGoalState(parent,explored,visited)
        if (x,y) in dotsList: #handle the situation when (x,y) is a dot in the list
            explored.add((cost,(x,y)))
            dotsList.remove((x,y)) #remove the dot from the list
        if (x,y) in wallList: #if the node is a wall, continue the loop
            continue
        visited.append((x,y))
        for i in [(x-1,y),(x+1,y),(x,y-1),(x,y+1)]:
            new_eval = cost + heuristic(x,y,dotsList,explored)+sum
            heapq.heappush(priority_queue, (new_eval,cost+1,(i[0],i[1]),(i[0],i[1],parent)))
        print (explored)
    return temp



def infoToGoalState(nodeList,explored ,visited):
    path = [] # create a null list
    while (nodeList != None):
        path.append((nodeList[0],nodeList[1]))  #repeat this step to add nodes into the path list.
        nodeList = nodeList[2]
    numOfExpandedNodes = len(visited)
    numOfExploredDots = len(explored)
    return [path, numOfExploredDots, numOfExpandedNodes]
    '''return the information about the multiple dots search,
     which includes the path to the goal state, the number of dots that have been visited
     and the number of expanded dots to the goal state'''

def manhatthanDist(node, otherNode):
    manhattan = abs(node[0] - otherNode[0]) + abs(node[1] - otherNode[1])
    return manhattan

def plotPath(result,maze,name):
    path = result[0]
    path = path[::-1]
    #print (path)
    orderList= list(range(10)) + list(string.ascii_lowercase) +list(string.ascii_uppercase)
    count = 0
    pathDict={}
    for i in path:
        pathDict[i] = orderList[count]
        count+=1

    for i in range(len(maze)):
        for j in range(len(maze[i])):
            if ((i,j) in path):
                maze[i][j] = pathDict[(i,j)]

    with open(str(name),'w') as f: # open a new file
        for i in range(len(maze)):
            for j in range(len(maze[i])):
                f.write(str(maze[i][j])) # write each element in the list into a file.
            f.write('\n')
        f.write("The total number of dots in the path is: "+ str(result[1]))
        f.write('\n')
        f.write("The number of expanded nodes is: " + str(result[2]))
        f.write('\n')
    #f.write("The total number of moves in the path is: " + len(path))
    #f.write('\n')

def main():
    maze = read_data("tinySearch.txt")
    dotsList = findAllDots(maze)

    for i in range(len(maze)):
        for j in range(len(maze[i])):
            if (maze[i][j] == 'P'):
                x = i
                y = j
    result = aStarMultiDots(x, y, maze)
    print (result)
    plotPath(result,maze,'result_tinySearch_test.txt')

def test():
    dotlist=[(1, 1), (1, 5), (1, 6), (3, 4), (3, 7), (5, 1), (5, 3), (5, 4), (5, 7), (7, 1), (7, 3), (7, 6), (7, 7)]
    result = heuristic(4,4,dotlist)
    print (result)


def mst(start,dotsList):
    actualExpanded = 0
    total = 0
    x = start[0]
    y = start[1]
    if len(dotsList) == 0:
        return 0

    priority_queue = []
    for node in dotsList:
        manDist = manhatthanDist(x,y,node[0],node[1])
        total += manDist
        heapq.heappush(priority_queue,(total,((x,y),node)))
    nodePair = heapq.heappop(priority_queue)
    return (nodePair[1][1],actualExpanded)


if __name__=='__main__':
    main()