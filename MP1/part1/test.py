import collections
import heapq
import string

def t():
    p_q =[]
    heapq.heappush(p_q,(4,(99,100,None)))
    print (p_q)
    heapq.heappush(p_q,(2,(95,98,None)))
    print (p_q)
    heapq.heappush(p_q,(10,(92,91,None)))
    heapq.heappush(p_q,(1,(11127,28,(12,21,(223,23,None)))))
    print(p_q)
    node = heapq.heappop(p_q)
    print(node)

def t1():
    priority_queue = []
    dotlist=[(1, 1), (1, 5), (1, 6), (3, 4), (3, 7), (5, 1), (5, 3), (5, 4), (5, 7), (7, 1), (7, 3), (7, 6), (7, 7)]
    edgesDict = {}
    for node in dotlist:
        for otherNode in dotlist:
            if (node == otherNode):
                continue
            manDist = manhatthanDist(node, otherNode)
            edgesDict[(node, otherNode)] = manDist
            heapq.heappush(priority_queue, (manDist,(node,otherNode))) # push the tuple into the prioprity queue
    nodeList = heapq.heappop(priority_queue)
    #print (nodeList)
    #print(edgesDict[nodeList[1]])
    result=closestDot(4,4,dotlist)
    print(result)
    orderList =list(range(10))
    orderList= list(range(10)) + (list(string.ascii_lowercase)) +(list(string.ascii_uppercase))
    print (orderList)

def closestDot(x, y, dotsList):
    edgesDict = {}  # use a dictionary to do the query in the next steps
    # explored = set()
    priority_queue = []
    temp = ()
    if (x, y) not in dotsList:
        dotsList.append((x, y))

    for node in dotsList:
        for otherNode in dotsList:
            if (node == otherNode):
                continue
            manDist = manhatthanDist(node, otherNode)  # calculate the manhattan distance between two dots
            edgesDict[(node, otherNode)] = manDist  # add the key-value pair into the dictionary
            heapq.heappush(priority_queue, (manDist, (node, otherNode)))  # push the tuple into the prioprity queue

    while len(priority_queue) > 0:
        nodeList = heapq.heappop(priority_queue)  # get the pair of nodes -> (manDist, (x1,y1),(x2,y2))
        firstNode = nodeList[1][0]
        secondNode = nodeList[1][1]
        if (firstNode == (x, y)):
            return (secondNode)
        if (secondNode == (x, y)):
            return (firstNode)
    return temp

def manhatthanDist(node, otherNode):
    manhattan = abs(node[0] - otherNode[0]) + abs(node[1] - otherNode[1])
    return manhattan
'''
class t2():
    import heapq
    import collections
    import string

    def read_data(name):
        with open(str(name), 'r') as f:
            lines = f.readlines()
            maze = []
            for i in range(len(lines)):
                maze.append(list(lines[i].strip()))

        # print((maze[20]))
        return maze

    def findAllDots(maze):
        # pass
        dotsList = []
        for i in range(len(maze)):
            for j in range(len(maze[i])):
                if (maze[i][j] == '.'):
                    dotsList.append((i, j))
        return dotsList

    def heuristic(dotsList):
        # pass
        # dotsList = findAllDots(maze) # get the list of the coordinates of dots
        edgesDict = {}  # use a dictionary to do the query in the next steps
        # explored = set()
        priority_queue = []
        # dotsList.append((x,y))
        for node in dotsList:
            for otherNode in dotsList:
                if (node == otherNode):
                    continue
                manDist = manhatthanDist(node, otherNode)  # calculate the manhattan distance between two dots
                edgesDict[(node, otherNode)] = manDist  # add the key-value pair into the dictionary
                heapq.heappush(priority_queue, (manDist, (node, otherNode)))  # push the tuple into the prioprity queue

        #sum the cost of each edge in the minimum spanning tree
        sum = 0
        while len(priority_queue) > 0:
            nodeList = heapq.heappop(priority_queue)  # get the pair of nodes -> (manDist, (x1,y1),(x2,y2))
            firstNode = nodeList[1][0]
            secondNode = nodeList[1][1]
            nodesPair = nodeList[1]
            # if (firstNode not in explored) or (secondNode not in explored):
            sum = sum + edgesDict[nodesPair]  # get the sum of all possible manDist between two dots

        return sum
    def closestDot(x, y, dotsList, first=False):
        edgesDict = {}
        # explored = set()
        priority_queue = []
        temp = ()
        if first:
            dotsList.append((x, y))
        for node in dotsList:
            for otherNode in dotsList:
                if (node == otherNode):
                    continue
                manDist = manhatthanDist(node, otherNode)  # calculate the manhattan distance between two dots
                edgesDict[(node, otherNode)] = manDist  # add the key-value pair into the dictionary
                heapq.heappush(priority_queue, (manDist, (node, otherNode)))  # push the tuple into the prioprity queue
        if first:
            dotsList.remove((x, y))
        while len(priority_queue) > 0:
            nodeList = heapq.heappop(priority_queue)  # get the pair of nodes -> (manDist, (x1,y1),(x2,y2))
            firstNode = nodeList[1][0]
            secondNode = nodeList[1][1]
            if (firstNode == (x, y)):
                return secondNode
            if (secondNode == (x, y)):
                return firstNode
        return temp

    def aStarMultiDots(x, y, maze):
        count = 0
        explored = set()
        visited = set()
        temp = []
        cost = 0
        dotsList = findAllDots(maze)
        priority_queue = []
        nextDot = closestDot(x, y, dotsList, first=True)
        evalValue = cost + heuristic(dotsList) + manhatthanDist((x, y), (nextDot[0], nextDot[1]))

        heapq.heappush(priority_queue, (evalValue, cost, (x, y, None)))


        while len(priority_queue) > 0:
            count += 1
            nodeList = heapq.heappop(priority_queue)
            x = nodeList[2][0]
            y = nodeList[2][1]
            cost = nodeList[1]

            if (len(dotsList) == 0):
                # goal state
                return infoToGoalState(nodeList[2], explored, count)

            # if ((x,y) in explored):
            #    continue
            if (maze[x][y] == '.'):  # when the node is a dot
                nextDot = closestDot(x, y, dotsList)
                dotsList.remove((x, y))  # remove the dot from the dotsList
                explored.add((x, y))  # add the dot into the explored set
                maze[x][y] = ' '
                if (len(dotsList) == 0):
                    # goal state
                    return infoToGoalState(nodeList[2], explored, count)

            if ((x, y) in visited):  # handle the repeated situation
                continue

            if (maze[x][y] == '%'):  # the situation when the node is not path
                visited.add((x, y))  # indicate the wall has been visited
                continue
            # visited.append((x, y))
            for i in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:  # move up, down, left and right
                new_heuristic = heuristic(dotsList)
                new_eval = cost + new_heuristic + manhatthanDist((x, y), (nextDot[0], nextDot[1]))
                heapq.heappush(priority_queue, (new_eval, cost + 1, (i[0], i[1], nodeList[2])))

            print(nodeList)

        return temp

    def infoToGoalState(nodeList, explored, count):
        path = []  # create a null list
        while (nodeList != None):
            path.append((nodeList[0], nodeList[1]))  # recurvisely add node into the path list.
            nodeList = nodeList[2]
        numOfExpandedNodes = count
        numOfExploredDots = len(explored)
        return [path, numOfExploredDots, numOfExpandedNodes]

    def manhatthanDist(node, otherNode):
        manhattan = abs(node[0] - otherNode[0]) + abs(node[1] - otherNode[1])
        return manhattan

    def plotPath(result, maze, name):
        path = result[0]
        path = path[::-1]
        # print (path)
        orderList = list(range(10)) + list(string.ascii_lowercase) + list(string.ascii_uppercase)
        count = 0
        pathDict = {}
        for i in path:
            pathDict[i] = orderList[count]
            count += 1

        for i in range(len(maze)):
            for j in range(len(maze[i])):
                if ((i, j) in path):
                    maze[i][j] = pathDict[(i, j)]

        with open(str(name), 'w') as f:  # open a new file
            for i in range(len(maze)):
                for j in range(len(maze[i])):
                    f.write(str(maze[i][j]))  # write each element in the list into a file.
                f.write('\n')
            f.write("The total number of dots in the path is: " + str(result[1]))
            f.write('\n')
            f.write("The number of expanded nodes is: " + str(result[2]))
            f.write('\n')
            # f.write("The total number of moves in the path is: " + len(path))
            # f.write('\n')

    def main():
        maze = read_data("smallSearch.txt")
        dotsList = findAllDots(maze)
        for i in range(len(maze)):
            for j in range(len(maze[i])):
                if (maze[i][j] == 'P'):
                    x = i
                    y = j
        print((x, y))
        result = aStarMultiDots(x, y, maze)
        print(result)
        plotPath(result, maze, 'result_smallSearch.txt')


    #if __name__ == '__main__':
     #   main()

'''
def t3():
    tempList=[0,1,2,3]
    '''temp = tempList[::2]  # [0,2]
    temp1 = tempList[1::2]  # [1,3]
    tempList = temp[::-1] + temp1[::-1]  # [2,0,3,1]
    tempList = tempList[::-1] # [1,3,0,2]
    tempNum=tempList[1]
    tempList[1]=tempList[3]
    tempList[3]=tempNum
'''
    queue = collections.deque(tempList)
    #temp = queue.pop()
    #queue.insert(0,temp)
    temp = queue.popleft()
    queue.append(temp)

    #result = [list.append(j) for i in tempList for j in tempList[i]]
    #print (temp)
    print (tempList)
    print(list(queue))
    tempp ={5:[0,1],6:[2,3]}
    print (tempp)
    for i in tempp:
        val = tempp[i][0]
        print (val)

if __name__=='__main__':
    t3()