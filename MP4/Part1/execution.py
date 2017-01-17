import numpy as np
import sys
from pong import *
import pygame
from pygame.locals import *

# Number of frames per second
FPS = 400
# configuration of the display
WINDOWWIDTH = 12 * 50
WINDOWHEIGHT = 12 * 50
LINETHICKNESS = 10
PADDLESIZE = WINDOWHEIGHT * 0.2

# Set up the colours
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


# Draws the arena the game will be played in.
def drawArena():
    DISPLAYSURF.fill((0, 0, 0))
    # Draw outline of arena
    pygame.draw.rect(DISPLAYSURF, WHITE, ((0, 0), (WINDOWWIDTH, WINDOWHEIGHT)), LINETHICKNESS * 2)


# Draws the paddle
def drawPaddle(paddle):
    # Stops paddle moving too low
    if paddle.bottom > WINDOWHEIGHT - LINETHICKNESS:
        paddle.bottom = WINDOWHEIGHT - LINETHICKNESS
    # Stops paddle moving too high
    elif paddle.top < LINETHICKNESS:
        paddle.top = LINETHICKNESS
    # Draws paddle
    pygame.draw.rect(DISPLAYSURF, WHITE, paddle)


# draws the ball
def drawBall(ball):
    pygame.draw.rect(DISPLAYSURF, WHITE, ball)


# moves the ball returns new position
def moveBall(ball, next_x, next_y):
    ball.x = next_x
    ball.y = next_y
    return ball


def movePaddle(paddle, next_x, next_y):
    paddle.x = next_x
    paddle.y = next_y
    return paddle


def findMaxBounce(M, temp):
    return max(M, temp)


def updateBall(cur_s, a):
    cur_s[0] += cur_s[2]
    cur_s[1] += cur_s[3]
    cur_s[4] += a


def train(Q, N_sa, s, r, cur_s, cur_r, a):
    i = 0
    while i < 100000:
        while True:
            a = Q_learning_process(s, r, a, Q, N_sa, cur_s, cur_r)

            updateBall(s, a)
            if bounce(s):
                r = 1

            if terminal(s):
                r = -1
                cur_s = [0.5, 0.5, 0.03, 0.01, 0.4]
                # s = [0.5, 0.5, 0.03, 0.01, 0.4]
                i += 1
                print("i: ", i)
                break

                # Q_learning algorithm
                # r = cur_r


def output(res):
    f = open('result.txt', 'w')
    temp = sum(res) / len(res)
    f.write('The average of 1000 test cases is ' + str(temp) + '\n')
    f.close()


def main():
    # initialization
    Q = [[[[[[0 for i5 in range(3)] for i4 in range(12)] for i3 in range(3)] for i2 in range(2)] for i1 in range(12)]
         for i0 in range(12)]
    N_sa = [[[[[[0 for j5 in range(3)] for j4 in range(12)] for j3 in range(3)] for j2 in range(2)] for j1 in range(12)]
            for j0 in range(12)]
    Q = np.array(Q)

    # define initial state, action and reward (all zeros)
    s = [0.5, 0.5, 0.03, 0.01, 0.4]
    a = 0
    r = 0

    cur_s = [0.5, 0.5, 0.03, 0.01, 0.4]
    cur_r = 0

    # define the necessary parameters:
    # num_bounce: total number of consecutive bounce on the paddle
    # num_learning: number of learning through each of the pong games
    # temp_bounce: temp num of the current bounce, if it is bigger than num_bounce, update it
    num_bounce = 0
    num_learning = 0
    temp_bounce = 0

    # first train the paddle for 100,000 times
    # train(Q,N_sa,s,r,cur_s,cur_r,a)
    '''
    j = 0

    while True:
        if j == 100000:
            break
        a = Q_learning_process(s, r, a, Q, N_sa, cur_s, cur_r)
        r = 0
        updateBall(cur_s, a)
        if bounce(s):
            r = 1

        if terminal(s):
            r = -1
            cur_s = [0.5, 0.5, 0.03, 0.01, 0.4]
            # s = [0.5, 0.5, 0.03, 0.01, 0.4]
            j += 1
            print("j: ", j)

    '''
    # initialization of the GUI
    pygame.init()
    global DISPLAYSURF
    ##Font information
    global BASICFONT, BASICFONTSIZE

    BASICFONTSIZE = 20
    BASICFONT = pygame.font.Font('freesansbold.ttf', BASICFONTSIZE)

    FPSCLOCK = pygame.time.Clock()
    DISPLAYSURF = pygame.display.set_mode((WINDOWWIDTH, WINDOWHEIGHT))
    pygame.display.set_caption('Pong')

    # Initiate variable and set starting positions
    # any future changes made within rectangles
    ballX = s[0] * (WINDOWWIDTH - LINETHICKNESS)
    ballY = s[1] * (WINDOWHEIGHT - LINETHICKNESS)
    playerOnePosition = s[4] * (WINDOWHEIGHT - PADDLESIZE)
    # playerTwoPosition = (WINDOWHEIGHT - PADDLESIZE) / 2
    # score = 0


    # Creates Rectangles for ball and paddles.
    paddle1 = pygame.Rect(WINDOWWIDTH - LINETHICKNESS, playerOnePosition, LINETHICKNESS, PADDLESIZE)
    # paddle2 = pygame.Rect(WINDOWWIDTH - PADDLEOFFSET - LINETHICKNESS, playerTwoPosition, LINETHICKNESS, PADDLESIZE)
    ball = pygame.Rect(ballX, ballY, LINETHICKNESS, LINETHICKNESS)

    # Draws the starting position of the Arena
    # drawArena()
    drawPaddle(paddle1)
    # drawPaddle(paddle2)
    drawBall(ball)

    res = []
    # pygame.mouse.set_visible(0) # make cursor invisible
    # watch out the reward update when bounce or terminate
    while True:  # main game loop
        if num_learning == 1000:
            break

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

        ballX = cur_s[0] * (WINDOWWIDTH - LINETHICKNESS)
        ballY = cur_s[1] * (WINDOWHEIGHT - LINETHICKNESS)
        playerOnePosition = cur_s[4] * (WINDOWHEIGHT - PADDLESIZE)

        # drawArena()
        drawPaddle(paddle1)
        # drawPaddle(paddle2)
        drawBall(ball)

        paddle1 = movePaddle(paddle1, WINDOWWIDTH - LINETHICKNESS, playerOnePosition)
        ball = moveBall(ball, ballX, ballY)
        # update the GUI
        pygame.display.update()
        DISPLAYSURF.fill(BLACK)

        updateBall(cur_s, a)

        num_bounce = findMaxBounce(num_bounce, temp_bounce)
        if bounce(cur_s):
            cur_r = 1
            temp_bounce += 1
            print("#bounce: ", temp_bounce)

            # temp_bounce = 0

        # do a terminal + reward check
        if terminal(cur_s):
            cur_r = -1
            cur_s = [0.5, 0.5, 0.03, 0.01, 0.4]
            # s = [0.5, 0.5, 0.03, 0.01, 0.4]
            num_learning += 1
            res.append(temp_bounce)
            temp_bounce = 0
            print("Max Bounce: ", num_bounce)

        # Q_learning algorithm
        a = Q_learning_process(s, r, a, Q, N_sa, cur_s, cur_r)
        r = cur_r

        FPSCLOCK.tick(FPS)
    output(res)


if __name__ == '__main__':
    main()
