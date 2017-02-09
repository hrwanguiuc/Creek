import copy

from CS440.MP1.part2.rubik import *

goalstate = [['r','r','r','r'],
        ['y','y','y','y'],
        ['g','g','g','g'],
        ['o','o','o','o'],
        ['b','b','b','b'],
        ['p','p','p','p']
]

goaltuple = ('r','r','r','r','y','y','y','y','g','g','g','g','o','o','o','o','b','b','b','b','p','p','p','p')


def heuristic(curstate,goalstate):
    diff = 0
    for i in range(6):
        for j in range(4):
            if curstate[i][j] != goalstate[i][j]:
                diff += 1
    #diff /= 8
    return diff


def checkstateequal(state1,state2):
    for i in range(6):
        for j in range(4):
            if state1[i][j] != state2[i][j]:
                return False
    return True


def printToFile(filename,graph):
    f = open(filename, 'w')
    f.write(graph[0][3] + ' ' + graph[0][0] + ' ' + graph[4][0] + ' '+ graph[4][1] + ' ' + graph[2][1] + ' ' + graph[2][2] + '\n')
    f.write(graph[0][2] + ' ' + graph[0][1] + ' ' + graph[4][3] + ' '+ graph[4][2] + ' ' + graph[2][0] + ' ' + graph[2][3] + '\n')
    f.write('    ' + graph[3][0] + ' ' + graph[3][1] + '\n')
    f.write('    ' + graph[3][3] + ' ' + graph[3][2] + '\n')
    f.write('    ' + graph[1][0] + ' ' + graph[1][1] + '\n')
    f.write('    ' + graph[1][3] + ' ' + graph[1][2] + '\n')
    f.write('    ' + graph[5][2] + ' ' + graph[5][3] + '\n')
    f.write('    ' + graph[5][1] + ' ' + graph[5][0] + '\n')
    f.close()

def readinputfile(filename):
    f = open(filename, 'r')
    graph = [['' for i in range(4)]for j in range(6)]
    for idx, line in enumerate(f.readlines()):
        if idx == 0:
            graph[0][3] ,graph[0][0] ,graph[4][0], graph[4][1] , graph[2][1] , graph[2][2] =line.split()
        if idx == 1:
            graph[0][2], graph[0][1] , graph[4][3] ,graph[4][2], graph[2][0], graph[2][3] = line.split()
        if idx == 2:
            graph[3][0], graph[3][1] = line.split()
        if idx == 3:
            graph[3][3], graph[3][2] = line.split()
        if idx == 4:
            graph[1][0], graph[1][1] = line.split()
        if idx == 5:
            graph[1][3], graph[1][2]  = line.split()
        if idx == 6:
            graph[5][2], graph[5][3] = line.split()
        if idx == 7:
            graph[5][1], graph[5][0]  = line.split()
    return graph

def Astar(cube):
    visited = {}   #initialize explored
    myq = []    #initialize queue
    myqhash = {}
    stategraph = cube.getGraph()
    startstate = State(0,cube,None,None)
    myq.append((startstate, heuristic(stategraph,goalstate))) #push to queue
    myqhash[startstate] = startstate

    while True:  #while the queue is not empty
       # print "q",myq
        #print "running"
        if len(myq) == 0:
            return None


        myq = sorted(myq, key = lambda t:t[1])
        (curstate,oldcost) = myq[0]     #pop from
       #print curstate.cube.getGraph(),oldcost
        myq.pop(0)
        curcube= curstate.cube
        #print len(myq) == 0
        #return

        if checkstateequal(curcube.getGraph(),goalstate):
    #        print curstate.cube.getconfig()
    #        print curstate.parent
    #        print curstate.move
            return curstate    # if reach the goal, return

        #else continue

        visited[curcube.getconfig()] = (curstate,oldcost)

        for i in range(6) :
            cwcube = copy.deepcopy(curcube)
            ccwcube = copy.deepcopy(curcube)

            cw(cwcube,i)

            if cwcube.getconfig() in visited:
            #    print "true"
                continue

           # nextcwstate = State(curstate.pathcost+1,cwcube,curcube.getconfig(),("cw",i))
            nextcwstate = State(curstate.pathcost+1,cwcube,curstate,("cw",i))
            tentativecost = heuristic(cwcube.getGraph(),goalstate)+nextcwstate.pathcost


            if cwcube.getconfig() in visited:
             #   print "running"
                continue
            else:
                flag = False
                for j in myq:
                    if cwcube.getconfig() == j[0].cube.getconfig():
                        if curstate.pathcost+1 <= j[0].pathcost:
                            myq.append((nextcwstate,tentativecost))
                            flag = True
                            break
                if not flag:
                    myq.append((nextcwstate,tentativecost))


            ccw(ccwcube,i)

            if ccwcube.getconfig() in visited:
             #   print "true"
                continue

            nextccwstate = State(curstate.pathcost+1,ccwcube,curstate,("ccw",i))
            tentativecost = heuristic(ccwcube.getGraph(),goalstate)+nextccwstate.pathcost


            if ccwcube.getconfig() in visited:
                continue
            else:
                flag = False
                for j in myq:
                    if ccwcube.getconfig() == j[0].cube.getconfig():
                        if curstate.pathcost+1 <= j[0].pathcost:
                            myq.append((nextccwstate,tentativecost))
                            flag = True
                            break
                if not flag:
                    myq.append((nextccwstate,tentativecost))





class State:
    def __init__(self,pathcost,cube,parentstate,move):
        self.pathcost = pathcost
        self.parent = parentstate
        self.cube = cube
        self.move = move



if __name__=='__main__':
    #newgraph = readinputfile("Input1.1.txt")
    #newgraph = readinputfile("Input1.2.txt")
    newgraph = readinputfile("Input1.3.txt")

    mycube = Cube(newgraph)

    startconfig = mycube.getconfig()

    finalstate = Astar(mycube)


    moveopration = []
    moves = []
    moves.append(finalstate.cube.getGraph)
    num = 1
    while finalstate.cube.getconfig() != startconfig:
        moves.append(finalstate.cube.getGraph())
        moveopration.append(finalstate.move)
        finalstate = finalstate.parent
        num +=1
        print (num)
    moves.append(newgraph)
    moves.reverse()
    moveopration.reverse()
    for i in range(num):
        printToFile("out1.3."+str(i)+".txt",moves[i])
    print (moveopration)
    #print moves




