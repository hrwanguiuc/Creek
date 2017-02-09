import numpy as np
import collections


#Final goal state set for each face
'''COLOR_SET ={'L':['r','r','r','r'],    #left
            'Bo':['b','b','b','b'],   #Bottom
            'R':['g','g','g','g'],    #Right
            'F':['o','o','o','o'],    #Front
            'T':['y','y','y','y'],    #Top
            'Ba':['p','p','p','p']}   #Back
'''
#Reference to face  #hashset
Face_To_Color ={0:'L',
            1:'Bo',
            2:'R',
            3:'F',
            4:'T',
            5:'Ba'}
#hash: face to number
Color_To_Face = {
            'L':0,
            'Bo':1,
            'R' :2,
            'F' :3,
            'T' :4,
            'Ba' :5
}
Color_Map = {'r':0,
             'y':1,
             'g':2,
             'o':3,
             'b':4,
             'p':5
}


GoalState = [[0,0,0,0],
        [1,1,1,1],
        [2,2,2,2],
        [3,3,3,3],
        [4,4,4,4],
        [5,5,5,5]
]


Inputstate1 = [['o','g','y','y'],
               ['g','p','p','g'],
               ['r','p','b','b'],
               ['b','b','g','o'],
               ['r','r','o','o'],
               ['y','y','p','r']
]


#             1
#      0   Tar_Face  2   # tarface at front
#             3



Adj = [[5,4,3,1], #adj of 0:L
       [0,3,2,5], #adj of 1:Bo
       [3,4,5,1], #adj of 2:R
       [0,4,2,1], #adj of 3:F
       [0,5,2,3], #adj of 4:T
       [2,4,0,1]  #adj of 5: Ba #? 0,1,2,4?
]



class Face:

    def __init__(self, face_index,inputcolor):
        self.num = face_index
        self.colorArr = inputcolor[face_index]
        self.adjfaces = Adj[face_index]
        self.adjindices = [0,0]*4
        if face_index == 0 :
            self.adjindices = [[2,1],[0,3],[0,3],[0,3]]
        if face_index == 1 :
            self.adjindices = [[3,2],[3,2],[3,2],[3,2]]
        if face_index == 2 :
            self.adjindices = [[2,1],[2,1],[0,3],[2,1]]
        if face_index == 3 :
            self.adjindices = [[2,1],[3,2],[0,3],[1,0]]
        if face_index == 4 :
            self.adjindices = [[1,0],[1,0],[1,0],[1,0]]
        if face_index == 5 :
            self.adjindices = [[2,1],[1,0],[0,3],[3,2]]



class simpleCube:
    def __init__(self,input):
        for i in range(6):
            self.graph[i] = input[i]

    def getGraph(self):
        return self.graph

class Cube:
    def __init__(self,input):
        self.face = [Face(0,input),Face(1,input),Face(2,input),Face(3,input),Face(4,input),Face(5,input)]

    def getGraph(self):
        graph = [[0 for i in range(4) ] for j in range(6)]
        for i in range (0,6):
            graph[i] = self.face[i].colorArr
        return graph

    def getconfig(self):
        array = []
        for i in range(6):
            array.extend(self.face[i].colorArr)
        return tuple(array)


def matchInput():    #map coolor char to int
    graph = [[0 for i in range(4) ] for j in range(6)]
    for i in range(6):
        for j in range(4):
            graph[i][j] = Color_Map.get(Inputstate1[i][j])
    return graph



def cw(cube,index):
    curface = cube.face[index]
    # rotate self face
    temp = curface.colorArr[3]
    for i in range(2,-1,-1):
        curface.colorArr[i+1] = curface.colorArr[i]
    curface.colorArr[0] = temp

    current_color = ['a' for i in range(8)]
    for i in range(0,4):
        for j in range (0,2):
            current_color[2*i+j] = cube.face[curface.adjfaces[i]].colorArr[curface.adjindices[i][j]]

    #print "curcolor",current_color
    tempcolor = [current_color[6],current_color[7]]
    #print "tempcolor",tempcolor

    for i in range(5,-1,-1):
        current_color[i+2] = current_color[i]

    #print current_color
    current_color[0] = tempcolor[0]
    current_color[1] = tempcolor[1]
    #print current_color
    for i in range(0,4):
        for j in range (0,2):
            cube.face[curface.adjfaces[i]].colorArr[curface.adjindices[i][j]] = current_color[2*i+j]



def ccw(cube,index):
    curface = cube.face[index]
    # rotate self face
    temp = curface.colorArr[0]
    for i in range (0,3):
        curface.colorArr[i] = curface.colorArr[i+1]
    curface.colorArr[3] = temp

    current_color = ['a' for i in range(8)]
    for i in range(0,4):
        for j in range (0,2):
            current_color[2*i+j] = cube.face[curface.adjfaces[i]].colorArr[curface.adjindices[i][j]]

    tempcolor = [current_color[0],current_color[1]]

    for i in range(0,6):
        current_color[i] = current_color[i+2]

    current_color[6] = tempcolor[0]
    current_color[7] = tempcolor[1]

    for i in range(0,4):
        for j in range (0,2):
            cube.face[curface.adjfaces[i]].colorArr[curface.adjindices[i][j]] = current_color[2*i+j]



if __name__=='__main__':
    #input = matchInput()   c


    mycube = Cube(Inputstate1)
    print (mycube.getGraph())

    for i in range(3):
        cw(mycube,0)
    print (mycube.getGraph())
    for i in range(3):
        ccw(mycube,0)
    print (mycube.getGraph())
    print (mycube.getconfig())

#          0 4 2          L  T  R
#            3               F
#            1               Bo
#            5               Ba
#
#
#


#      3,0  0,1  1,2
#      2,1  3,2  0,3
#           0,1
#           3,2
#           0,1
#           3,2
#           2,3
#           1,0







#Suppose each face the label is |0,1| i.e. from left to right, top to bottom
#                               |3,2|
#
