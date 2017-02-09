import collections
import heapq
import copy
#
#Suppose each face the label is |0,1| i.e. from left to right, top to bottom
#                               |2,3|
#


#Final goal state set for each face
COLOR_SET ={'L':['r','r','r','r'],
            'U':['b','b','b','b'],
            'R':['g','g','g','g'],
            'F':['o','o','o','o'],
            'D':['y','y','y','y'],
            'B':['p','p','p','p']}
#Reference to color set
REF_COLOR ={0:'L',
            1:'U',
            2:'R',
            3:'F',
            4:'D',
            5:'B'}

class face(object):
    def __init__(self,colorArr,adj): # color array and adj array colorArr = 6x4 adj = 4x2

        self.colorArr = colorArr
        self.adj = adj
        #self.num = num

    def get_colorArr(self,index):
        return (self.colorArr)[index]

    def set_colorArr(self,index,value):
        (self.colorArr)[index] = value

    def get_adj(self):
        return self.adj

    #def set_num(self,num):
        self.num = num


class Cube(object):
    faceDict={}
    def __init__(self,colorArray):
        self.colorArray= colorArray
        l = face(colorArray[0],[1,3,4,5]) # the adjacent is formed as clockwise in this structure
        u = face(colorArray[1],[0,5,2,3])
        r = face(colorArray[2],[1,5,4,3])
        f = face(colorArray[3],[0,1,2,4])
        d = face(colorArray[4],[0,3,2,5])
        b = face(colorArray[5],[0,4,2,1])

        self.faceDict = {0: l, 1: u, 2: r, 3: f, 4: d, 5: b}
        self.stateTable =[]
        count = 0
        #self.refTable=['l','d','r','f','u','b']
        #for i in self.refTable:
         #   k = self.refTable[count]
        #self.stateTable.append((self.faceDict[k]).get_colorArr()) # state table for goal state test
        #    count+=1
        for i in self.faceDict:
            self.stateTable.append(self.faceDict[count].colorArr)
            count+=1

        # Create 8 cubies to use for heuristic function
        # The order is also: |0,1| from the top view
        #                    |2,3|
        self.cubies = []
        self.cubies.append([b.colorArr[1],l.colorArr[0],u.colorArr[0]])
        self.cubies.append([b.colorArr[0],r.colorArr[1],u.colorArr[1]])
        self.cubies.append([f.colorArr[0],l.colorArr[1],u.colorArr[2]])
        self.cubies.append([f.colorArr[1],r.colorArr[0],u.colorArr[3]])
        self.cubies.append([f.colorArr[2],l.colorArr[3],d.colorArr[0]])
        self.cubies.append([f.colorArr[3],r.colorArr[2],d.colorArr[1]])
        self.cubies.append([b.colorArr[3],l.colorArr[2],d.colorArr[2]])
        self.cubies.append([b.colorArr[2],r.colorArr[3],d.colorArr[3]])

    def get_cubies(self):
        return self.cubies

    def get_face(self,index):
        temp = self.faceDict[index]
        return temp
    def getState(self):
        tempState = []
        for i in range(6):
            tempState.append((self.faceDict[i]).colorArr)
        return tempState  #return the current state 4x2 matrix


def rotate_face(tempList, ccw=False):
    if (ccw):
        temp = tempList[::2]  # [0,2]
        temp1 = tempList[1::2]  # [1,3]
        tempList = temp[::-1] + temp1[::-1]  # [2,0,3,1]
        tempList = tempList[::-1]  # [1,3,0,2]


        return tempList # [1,3,0,2]
    else:
        temp = tempList[::2]  # [0,2]
        temp1 = tempList[1::2]  # [1,3]
        tempList = temp[::-1] + temp1[::-1]  # [2,0,3,1]

        return tempList

def rotate_adj(adjList, ccw=False):
    valueList = []
    tempList = []

    for i in range(4):
        valueList.append(adjList[i])

    queue = collections.deque(valueList)  # valueList is a 4x2 matrix
    if ccw:
        temp = queue.popleft()
        queue.append(temp)

    else:
        temp = queue.pop()
        queue.insert(0, temp)

    valueList = list(queue)
    #print (valueList)
    return valueList

def rotate(cube,dire,ccw=False):
    # pass
    if (dire == 'U'):
        tempList = {}
        tempList1=[]
        tempFace = cube.faceDict[1]
        adjList = tempFace.get_adj()

        for i in adjList:
            #tempList[i] = ([(cube.faceDict[i]).colorArr[0], (cube.faceDict[i]).colorArr[1]])  # 4x2 matrix
            tempList1.append([cube.colorArray[i][0],cube.colorArray[i][1]])

        if (not ccw):
            changedList = rotate_adj(tempList1, adjList)  # adjacent finish
            tempFace.colorArr = rotate_face(tempFace.colorArr)  # face finish

        if (ccw):
            changedList = rotate_adj(tempList1, ccw=True)  # ccw case
            tempFace.colorArr = rotate_face(tempFace.colorArr, ccw=True)
        # change the mapped value in cube
        count = 0
        for i in adjList:
            #(cube.faceDict[i]).colorArr[0] = (changedList[i])[0]
            #(cube.faceDict[i]).colorArr[1] = (changedList[i])[1]
            cube.colorArray[i][0] = changedList[count][0]
            cube.colorArray[i][1] = changedList[count][1]
            count +=1
        cube.faceDict[1] = tempFace

    if (dire == 'D'):
        tempList = {}
        tempList1=[]
        tempFace = cube.faceDict[4]
        adjList = tempFace.get_adj()

        for i in adjList:
            #tempList[i] = ([(cube.faceDict[i]).colorArr[3], (cube.faceDict[i]).colorArr[2]])  # 4x2 matrix
            tempList1.append([cube.colorArray[i][3], cube.colorArray[i][2]])

        if (not ccw):
            changedList = rotate_adj(tempList1)  # adjacent finish
            tempFace.colorArr = rotate_face(tempFace.colorArr)  # face finish

        if (ccw):
            changedList = rotate_adj(tempList1, ccw=True)  # ccw case
            tempFace.colorArr = rotate_face(tempFace.colorArr, ccw=True)
        # change the mapped value in cube
        count =0
        for i in adjList:
            #(cube.faceDict[i]).colorArr[3] = (changedList[i])[0]
            #(cube.faceDict[i]).colorArr[2] = (changedList[i])[1]
            cube.colorArray[i][3] = changedList[count][0]
            cube.colorArray[i][2] = changedList[count][1]
            count+=1
        cube.faceDict[4] = tempFace


    if (dire == 'L'):
        tempList = {}
        tempList1 =[]
        tempFace = cube.faceDict[0]
        adjList = tempFace.get_adj()

        for i in adjList:
            if (i == 5):
                #tempList[i] = (
                #[(cube.faceDict[i]).colorArr[1], (cube.faceDict[i]).colorArr[3]])  # back side index should be 1,3
                tempList1.append([cube.colorArray[i][1],cube.colorArray[i][3]])

            else:
                #tempList[i] = ([(cube.faceDict[i]).colorArr[2],
                #                (cube.faceDict[i]).colorArr[0]])  # front, top and bottom index should be 2,0
                tempList1.append([cube.colorArray[i][2],cube.colorArray[i][0]])

        if (not ccw):
            changedList = rotate_adj(tempList1)  # adjacent finish
            tempFace.colorArr = rotate_face(tempFace.colorArr)  # face finish

        if (ccw):
            changedList = rotate_adj(tempList1, ccw=True)  # ccw case
            tempFace.colorArr = rotate_face(tempFace.colorArr, ccw=True)
        # change the mapped value in cube
        count =0
        for i in adjList:
            if (i == 5):
                #(cube.faceDict[i]).colorArr[1] = changedList[i][0]
                #(cube.faceDict[i]).colorArr[3] = changedList[i][1]
                cube.colorArray[i][1]=changedList[count][0]
                cube.colorArray[i][3]=changedList[count][1]
                count+=1

            else:
                #(cube.faceDict[i]).colorArr[2] = changedList[i][0]
                #(cube.faceDict[i]).colorArr[0] = changedList[i][1]
                cube.colorArray[i][2] = changedList[count][0]
                cube.colorArray[i][0] = changedList[count][1]
                count+=1

        cube.faceDict[0] = tempFace

    if (dire == 'R'):
        tempList = {}
        tempList1 =[]
        tempFace = cube.faceDict[2]
        adjList = tempFace.get_adj()

        for i in adjList:
            if (i == 5):
                #tempList[i] = (
                #[(cube.faceDict[i]).colorArr[2], (cube.faceDict[i]).colorArr[0]])  # back side index should be 2,0
                tempList1.append([cube.colorArray[i][2],cube.colorArray[i][0]])

            else:
                #tempList[i] = (
                #[(cube.faceDict[i]).colorArr[1], (cube.faceDict[i]).colorArr[3]])  # other side index should be 1,3
                tempList1.append([cube.colorArray[i][1], cube.colorArray[i][3]])
        if (not ccw):
            changedList = rotate_adj(tempList1)  # adjacent finish
            tempFace.colorArr = rotate_face(tempFace.colorArr)  # face finish

        if (ccw):
            changedList = rotate_adj(tempList1, ccw=True)  # ccw case
            tempFace.colorArr = rotate_face(tempFace.colorArr, ccw=True)
        # change the mapped value in cube
        # print (tempList)
        # print (changedList)
        count=0
        for i in adjList:
            if (i == 5):
                #(cube.faceDict[i]).colorArr[2] = changedList[i][0]  # back index should be 2,0
                #(cube.faceDict[i]).colorArr[0] = changedList[i][1]
                cube.colorArray[i][2]=changedList[count][0]
                cube.colorArray[i][0] = changedList[count][1]
                count+=1

            else:
                #(cube.faceDict[i]).colorArr[1] = changedList[i][0]
                #(cube.faceDict[i]).colorArr[3] = changedList[i][1]
                cube.colorArray[i][1] = changedList[count][0]
                cube.colorArray[i][3] = changedList[count][1]
                count+=1

        cube.faceDict[2] = tempFace

    if (dire == 'F'):
        tempList = {}
        tempList1 =[]
        tempFace = cube.faceDict[3]
        adjList = tempFace.get_adj()

        for i in adjList:
            if (i == 4):
                #tempList[i] = (
                #[(cube.faceDict[i]).colorArr[0], (cube.faceDict[i]).colorArr[1]])  # bottom side index should be 0,1
                tempList1.append([cube.colorArray[i][0],cube.colorArray[i][1]])
            if (i == 2):
                #tempList[i] = (
                #[(cube.faceDict[i]).colorArr[2], (cube.faceDict[i]).colorArr[0]])  # right side index should be 2,0
                tempList1.append([cube.colorArray[i][2],cube.colorArray[i][0]])

            if (i == 0):
                #tempList[i] = (
                #[(cube.faceDict[i]).colorArr[1], (cube.faceDict[i]).colorArr[3]])  # left side index should be 1,3
                tempList1.append([cube.colorArray[i][1],cube.colorArray[i][3]])

            if (i == 1):
                #tempList[i] = (
                #[(cube.faceDict[i]).colorArr[3], (cube.faceDict[i]).colorArr[2]])  # top side index should be 3,2
                tempList1.append([cube.colorArray[i][3],cube.colorArray[i][2]])


        if (not ccw):
            changedList = rotate_adj(tempList1)  # adjacent finish
            tempFace.colorArr = rotate_face(tempFace.colorArr)  # face finish

        if (ccw):
            changedList = rotate_adj(tempList1, ccw=True)  # ccw case
            tempFace.colorArr = rotate_face(tempFace.colorArr, ccw=True)
        # change the mapped value in cube

        count =0
        for i in adjList:
            if (i == 4):
                #(cube.faceDict[i]).colorArr[0] = changedList[i][0]
                #(cube.faceDict[i]).colorArr[1] = changedList[i][1]
                cube.colorArray[i][0] = changedList[count][0]
                cube.colorArray[i][1] = changedList[count][1]
                count+=1
            if (i == 2):
                #(cube.faceDict[i]).colorArr[2] = changedList[i][0]
                #(cube.faceDict[i]).colorArr[0] = changedList[i][1]
                cube.colorArray[i][2] = changedList[count][0]
                cube.colorArray[i][0] = changedList[count][1]
                count+=1
            if (i == 0):
                #(cube.faceDict[i]).colorArr[1] = changedList[i][0]
                #(cube.faceDict[i]).colorArr[3] = changedList[i][1]
                cube.colorArray[i][1] = changedList[count][0]
                cube.colorArray[i][3] = changedList[count][1]
                count+=1
            if (i == 1):
                #(cube.faceDict[i]).colorArr[3] = changedList[i][0]
                #(cube.faceDict[i]).colorArr[2] = changedList[i][1]
                cube.colorArray[i][3] = changedList[count][0]
                cube.colorArray[i][2] = changedList[count][1]
                count+=1

        cube.faceDict[3] = tempFace

    if (dire == 'B'):
        tempList = {}
        tempList1=[]
        tempFace = cube.faceDict[5]
        adjList = tempFace.get_adj()

        for i in adjList:
            if (i == 4):
                #tempList[i] = ([(cube.faceDict[i]).colorArr[3],
                #                (cube.faceDict[i]).colorArr[2]])  # bottom side index should be 3,2
                tempList1.append([cube.colorArray[i][3],cube.colorArray[i][2]])
            if (i == 2):
                #tempList[i] = ([(cube.faceDict[i]).colorArr[1],
                #                (cube.faceDict[i]).colorArr[3]])  # right side index should be 1,3
                tempList1.append([cube.colorArray[i][1], cube.colorArray[i][3]])
            if (i == 0):
                #tempList[i] = ([(cube.faceDict[i]).colorArr[2],
                #                (cube.faceDict[i]).colorArr[0]])  # left side index should be 2,0
                tempList1.append([cube.colorArray[i][2], cube.colorArray[i][0]])
            if (i == 1):
                #tempList[i] = ([(cube.faceDict[i]).colorArr[0],
                #                (cube.faceDict[i]).colorArr[1]])  # top side index should be 0,1
                tempList1.append([cube.colorArray[i][0], cube.colorArray[i][1]])

        if (not ccw):
            changedList = rotate_adj(tempList1)  # adjacent finish
            tempFace.colorArr = rotate_face(tempFace.colorArr)  # face finish

        if (ccw):
            changedList = rotate_adj(tempList1, ccw=True)  # ccw case
            tempFace.colorArr = rotate_face(tempFace.colorArr, ccw=True)
        # change the mapped value in cube
        count =0
        for i in adjList:
            if (i == 4):
                #(cube.faceDict[i]).colorArr[3] = changedList[i][0]
                #(cube.faceDict[i]).colorArr[2] = changedList[i][1]
                cube.colorArray[i][3]=changedList[count][0]
                cube.colorArray[i][2]=changedList[count][1]
                count+=1
            if (i == 2):
                #(cube.faceDict[i]).colorArr[1] = changedList[i][0]
                #(cube.faceDict[i]).colorArr[3] = changedList[i][1]
                cube.colorArray[i][1] = changedList[count][0]
                cube.colorArray[i][3] = changedList[count][1]
                count+=1

            if (i == 0):
                #(cube.faceDict[i]).colorArr[2] = changedList[i][0]
                #(cube.faceDict[i]).colorArr[0] = changedList[i][1]
                cube.colorArray[i][2] = changedList[count][0]
                cube.colorArray[i][0] = changedList[count][1]
                count+=1

            if (i == 1):
                #(cube.faceDict[i]).colorArr[0] = changedList[i][0]
                #(cube.faceDict[i]).colorArr[1] = changedList[i][1]
                cube.colorArray[i][0] = changedList[count][0]
                cube.colorArray[i][1] = changedList[count][1]
                count+=1

        cube.faceDict[5] = tempFace

    tempArr = []
    for i in range(6):
        tempArr.append((cube.faceDict[i]).colorArr)
    #print(tempArr)
    new_cube = Cube(tempArr)
    return new_cube

##########################################################





def goalState():
    count = 0
    goalState = []  # 6x4 matrix
    for i in COLOR_SET:
        goalState.append(COLOR_SET[REF_COLOR[count]])
        count += 1

    return goalState

def heuristic(cube,goal):

    tempState =[]
    for i in range(6):
        tempState.append((cube.faceDict[i]).colorArr)

    misplacedNum = 0

    for i in range(len(tempState)):
        for j in range(len(tempState[i])):
            if (tempState[i][j]!= goal[i][j]):
                misplacedNum+=1
    #print (misplacedNum)
    return misplacedNum

def aStar_cube(cube):
    visited = set()
    temp = []
    movements=('U','u','L','l','D','d','R','r','B','b','F','f') # lower case denotes ccw, upper case denotes cw
    cost = 0
    bestPath=None
    goal = goalState()
    print (goal)

    priority_queue = []
    heapq.heappush(priority_queue, ((cost + heuristic(cube,goal)), cost, cube.colorArray,('S')))
    '''push the start state with its heuristic value into the prioprity queue'''
    while (len(priority_queue) > 0):
        curr = heapq.heappop(priority_queue)
        curr_state = curr[2]
        tempCube = Cube(curr[2])
        cost = curr[1]
        seq = curr[3]
        curr_tuple = (curr_state,cost,seq[0]) # hash three elements including curr_state matrix, cost and movement sequence
        curr_state_hash = hash(str(curr_tuple)) # get a hash value of the element
        if (curr_state == goal):  # the goal state
            #print ("Movements: ",len(visited)) #total expanded movements in the search
            print("Path length is: ",len(pathToGoalState(seq)))
            if bestPath is None or (len(pathToGoalState(seq))<len(pathToGoalState(bestPath))):
                bestPath = pathToGoalState(seq)
            #return bestPath
        if (curr_state_hash in visited):  # handle the repeated situation
            continue
        visited.add(curr_state_hash)
        for i in movements:  # 12 possible turns
            temp = copy.deepcopy(tempCube)
            if (i == 'U'): # turn top 90 degrees, clockwise
                temp = rotate(temp,'U')
            elif (i == 'u'): # turn top 90 degrees, counterclockwise
                temp = rotate(temp,'U',ccw=True)
            elif (i == 'L'): # turn left 90 degrees, clockwise
                temp = rotate(temp, 'L')
            elif (i == 'l'):# turn left 90 degrees, counterclockwise
                temp = rotate(temp, 'L', ccw=True)
            elif (i == 'D'):
                temp = rotate(temp, 'D')
            elif (i == 'd'):# turn bottom 90 degrees, counterclockwise
                temp = rotate(temp, 'D', ccw=True)
            elif (i == 'R'):
                temp = rotate(temp, 'R')
            elif (i == 'r'):# turn right 90 degrees, counterclockwise
                temp = rotate(temp, 'R', ccw=True)
            elif (i == 'B'):
                temp = rotate(temp, 'B')
            elif (i == 'b'):# turn back 90 degrees, counterclockwise
                temp = rotate(temp, 'B', ccw=True)
            elif (i == 'F'):
                temp = rotate(temp, 'F')
            elif (i == 'f'):# turn front 90 degrees, counterclockwise
                temp = rotate(temp, 'F', ccw=True)

            new_eval = cost + heuristic(temp,goal)
            new_array = temp.colorArray
            heapq.heappush(priority_queue, (new_eval, cost+1, new_array, (i,seq)))
            '''combine the evaluation function value(cost+heuristic) of the current node and its parent as
            new node and add it to the priority queue'''
        #print(temp.colorArray)

    return bestPath

def bfs_cubie(cube,index,goal,colorArr):
    res=[]
    visited =set() # create a set to store visited node
    seq =[]
    cubie = (cube.get_cubies())[index]
    q = collections.deque((cubie,(cubie,None))) # create a queue
    movements=('U','u','L','l','D','d','R','r','B','b','F','f') # lower case denotes ccw, upper case denotes cw

    while len(q)>0:

        curr_node = q.popleft() # use popleft() to implement FIFO
        curr_cubie=curr_node[0]
        parent = curr_node[1]

        curr_hash = hash(cubie[0]+cubie[1]+cubie[2])
        if (curr_cubie == goal[index]): # the goal state

            return pathToGoalState(parent)
        if (curr_hash in visited):
            continue

        visited.add(curr_hash)# indicate the node has been visited

        for i in movements:  # move up, down, left and right
            temp = cube
            if (i == 'U'):  # turn top 90 degrees, clockwise
                new_cube = rotate(temp, 'U')
            elif (i == 'u'):# turn top 90 degrees, counterclockwise
                new_cube = rotate(temp,'U',ccw=True)
                print (new_cube.colorArray)
            elif (i == 'L'):  # turn left 90 degrees, clockwise
                new_cube = rotate(temp, 'L')
            elif (i == 'l'):  # turn left 90 degrees, counterclockwise
                new_cube = rotate(temp, 'L', ccw=True)
                temp = cube
            elif (i == 'D'):
                new_cube = rotate(temp, 'D')
                temp = cube
            elif (i == 'd'):  # turn bottom 90 degrees, counterclockwise
                new_cube = rotate(temp, 'D', ccw=True)
                temp = cube
            elif (i == 'R'):
                new_cube = rotate(temp, 'R')
            elif (i == 'r'):  # turn right 90 degrees, counterclockwise
                temp = cube
                new_cube = rotate(cube, 'R', ccw=True)
            elif (i == 'B'):
                new_cube = rotate(temp, 'B')
            elif (i == 'b'):  # turn back 90 degrees, counterclockwise
                new_cube = rotate(temp, 'B', ccw=True)
            elif (i == 'F'):
                new_cube = rotate(temp, 'F')
            elif (i == 'f'):  # turn front 90 degrees, counterclockwise
                new_cube = rotate(temp, 'F', ccw=True)

            curr_cubie = (new_cube.get_cubies())[index] # get the changed cubie
            print(curr_cubie)
            q.append((curr_cubie,new_cube,(curr_cubie,parent))) # combine the current cubie and parent as new node and add it to the queue
            #print(nodeList)
    return res



def pathToGoalState(seq):
    path = [] # create a null list
    while (seq!='S' and len(seq)>1):
        path.append(seq[0])  #recurvisely add node into the path list.
        seq = seq[1]
    return path # return a list of ('M') pairs, which is the path to the goal state
def test():
    pass

def organizer():
    colorArr = [['o','g','y','y'],
                ['r','r','o','o'],
                ['r','p','b','b'],
                ['b','b','o','g'],
                ['g','p','g','p'],
                ['y','y','r','p']]
    colorArr2= [['y','r','o','g'],
                ['g','b','o','r'],
                ['y','p','b','g'],
                ['y','p','o','g'],
                ['b','p','r','y'],
                ['r','o','p','b']]
    colorArr3= [['y','p','o','g'],
                ['o','g','r','b'],
                ['r','o','b','g'],
                ['y','p','o','g'],
                ['b','p','r','y'],
                ['y','r','p','b']]
    tempCube = Cube(colorArr3)
    #newCube = rotate(tempCube,'B')
    #print("After rotation:")
    #print(newCube.colorArray)
    result = aStar_cube(tempCube)
    print ("The movement sequence is: ",result)

if __name__=='__main__':
    organizer()
    #test()








