import math
import numpy as np
import random

# States = (ball_x, ball_y, velocity_x, velocity_y, paddle_y)
# Actions = ['up':0,'down':1,'stop':2]
# Rewards = ['bounce':1,'pass':-1,'o':0]
# V_DISCRETE is the discrete value of the velocity
ACTIONS = [0.04, -0.04, 0]
REWARDS = [-1, 0, 1]
V_DISCRETE = [-1, 0, 1]

INITIAL = (0.5, 0.5, 0.03, 0.01, 0.4)
cur_reward = 0

paddle_height = 0.2
paddle_x = 1

GAMMA = 0.9
C = 40
Ne = 100  # a fixed number for exploration
R_p = 50  # optimistic reward estimate
TRAIN_TIMES = 100000

# Q(s,a) = [ball_x][ball_y][velocity_x][velocity_y][paddle_y][action]
'''
Q = [[[[[[0 for i5 in range(3)]for i4 in range(12)] for i3 in range(3)] for i2 in range(2)] for i1 in range(12)] for i0 in range(12)]
N_sa = [[[[[[0 for j5 in range(3)]for j4 in range(12)] for j3 in range(3)] for j2 in range(2)] for j1 in range(12)] for j0 in range(12)]
Q = np.array(Q)
'''


def bounce(state):
    '''

    :param state: state representation [ball_x, ball_y, velocity_x, velocity_y, paddle_y]
    :return: True if it bounces at the paddle, False if it doesn't
    '''

    # totally four conditions to consider, watch out the update for velocity
    # bounce at the wall, left edge of the screen
    if state[0] < 0:
        state[0] = -state[0]
        state[2] = -state[2]
    # bounce at the top edge of the screen
    if state[1] < 0:
        state[1] = -state[1]
        state[3] = -state[3]
    # bounce at the bottom edge of the screen
    if state[1] > 1:
        state[1] = 2 - state[1]
        state[3] = -state[3]
    # bounce at the paddle, right edge of the screen
    if (state[0] >= 1) and (state[1] > state[4]) and (state[1] < state[4] + paddle_height):
        state[0] = 2 * paddle_x - state[0]
        ref = 1
        state[2] = -state[2]
        ori_vx = state[2]
        while ref > 0:
            state[2] = ori_vx + np.random.uniform(-0.015, 0.015)
            if abs(state[2]) > 0.03:
                state[3] = state[3] + np.random.uniform(-0.03, 0.03)
                ref = -1

        return True

    return False


def discretize(state):
    b_x = int(math.floor(12 * state[0]))
    if b_x > 11:
        b_x = 11
    b_y = int(math.floor(12 * state[1]))
    if b_y > 11:
        b_y = 11
    v_x = int(np.sign(state[2]))
    if abs(state[3]) < 0.015:
        v_y = 0
    else:
        v_y = int(np.sign(state[3]))
    if state[4] >= 1:
        p_y = 11
    elif state[4] < 0:
        p_y = 0
    else:
        p_y = int(math.floor(12 * state[4] / (1 - paddle_height)))
        if p_y > 11:
            p_y = 11

    return b_x, b_y, v_x, v_y, p_y


def discretize_action(a):
    # up:0, down:1, stop:2
    if a == 0.04:
        return 0
    elif a == -0.04:
        return 1
    else:
        return 2


def terminal(state):
    condition_1 = False
    condition_2 = False
    if (state[1] < state[4]) or (state[1] > state[4] + paddle_height):
        condition_1 = True
    if state[0] > paddle_x:
        condition_2 = True
    if condition_1 and condition_2:
        return True

    return False


def raw_exp(Q, N):
    temp = [0, 0, 0]
    for i in range(3):
        if N[i] < Ne:
            temp[i] = R_p
        else:
            temp[i] = Q[i]
    temp = np.array(temp)
    return np.argmax(temp)


def exploration(Q, N):
    temp = [0, 0, 0]
    for i in range(3):
        if N[i] < Ne:
            temp[i] = R_p
        else:
            temp[i] = Q[i]

    a = tie_breaker(temp)

    # print("Next action:",a)
    return a


def tie_breaker(exps):
    Q1 = exps[0]
    Q2 = exps[1]
    Q3 = exps[2]
    if Q1 == Q2 == Q3:
        return random.randint(0, 2)
    elif Q1 == Q2:
        return random.randint(0, 1)
    elif Q2 == Q3:
        return random.randint(1, 2)
    elif Q1 == Q3:
        temp_num = 1
        while (temp_num != 1):
            temp_num = random.randint(0, 2)
        return temp_num
    else:
        v = max([Q1, Q2, Q3])
        if v == Q1:
            return 0
        elif v == Q2:
            return 1
        elif v == Q3:
            return 2


def Q_learning_process(s, r, act, Q, N_sa, cur_s, cur_r):
    # execute the discretize process
    bx, by, vx, vy, py = discretize(s)
    cbx, cby, cvx, cvy, cpy = discretize(cur_s)
    a = discretize_action(act)
    # print("bx:",bx," by: ",by)
    Qs = Q[bx][by][vx][vy][py][a]
    # Qcs -> find the argmax of actions, so it is an array
    Qcs = Q[cbx][cby][cvx][cvy][cpy]
    N_cs = N_sa[cbx][cby][cvx][cvy][cpy]

    # deal with learning rate
    alpha = C / (C + N_sa[bx][by][vx][vy][py][a])

    # find the next action index
    max_a = np.argmax(Qcs)

    if terminal(s):
        r = -1
        N_sa[bx][by][vx][vy][py][a] += 1
        # update Q value according to TD learning rule
        # *(N_sa[bx][by][vx][vy][py][a])
        Q[bx][by][vx][vy][py][a] += alpha * \
                                    (r - Q[bx][by][vx][vy][py][a])
        # update s

    else:
        # update the N_sa
        N_sa[bx][by][vx][vy][py][a] += 1
        # update Q value according to TD learning rule
        # *(N_sa[bx][by][vx][vy][py][a])
        Q[bx][by][vx][vy][py][a] += alpha * \
                                    (r + GAMMA * Q[cbx][cby][cvx][cvy][cpy][max_a] - Q[bx][by][vx][vy][py][a])
        # update s
    for i in range(len(cur_s)):
        s[i] = cur_s[i]
    # update r

    # use exploration function to find the best action
    # then update a
    # exps = []

    # for i in range(3):
    #    exps.append(exploration(Qcs[i],N_cs[i]))
    # exps = np.array(exps)
    next_a_idx = exploration(Qcs, N_cs)
    # next_a_idx = raw_exp(Qcs, N_cs)
    # next_a_idx = np.argmax(exps)
    a = ACTIONS[next_a_idx]

    return a


# main execution of the TD training process
def main():
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

    t = 0
    while True:

        if t == TRAIN_TIMES:
            print("Training finished!")
            break

        cur_s[0] += cur_s[2]
        cur_s[1] += cur_s[3]
        cur_s[4] += a
        cur_r = 0
        # print("Before check: ",cur_s)
        if bounce(cur_s):
            print("bounce")
            cur_r = 1
        # print("After check: ", cur_s)
        # Q learning process
        if terminal(cur_s):
            # s = [0.5, 0.5, 0.03, 0.01, 0.4]
            # r = 0
            cur_s = [0.5, 0.5, 0.03, 0.01, 0.4]
            cur_r = 0
            # a = 0
            t += 1
            print("t: ", t)

        a = Q_learning_process(s, r, a, Q, N_sa, cur_s, cur_r)
        r = cur_r
    # test the agent

    s = [0.5, 0.5, 0.03, 0.01, 0.4]
    cur_s = [0.5, 0.5, 0.03, 0.01, 0.4]
    r = 0
    cur_r = 0
    a = 0
    print(Q)
    test(r, s, cur_s, cur_r, Q, N_sa, a)


def test(r, s, cur_s, cur_r, Q, N_sa, a):
    T = 0
    n_bounce = 0
    bounce_list = []

    while T < 1000:

        cur_s[0] += cur_s[2]
        cur_s[1] += cur_s[3]
        cur_s[4] += a
        cur_r = 0

        if bounce(s):
            r = 1
            n_bounce += 1
            print("Bounce: ")

        if terminal(cur_s):
            # r = 0
            # s = [0.5, 0.5, 0.03, 0.01, 0.4]
            cur_s = [0.5, 0.5, 0.03, 0.01, 0.4]
            # a = 0
            cur_r = 0
            T += 1
            print("test: ", T)
            print(n_bounce)
            bounce_list.append(n_bounce)
            n_bounce = 0

        a = Q_learning_process(s, r, a, Q, N_sa, cur_s, cur_r)

    print("---------------------------------------------------")
    print("total:", sum(bounce_list))
    print("len:", len(bounce_list))
    print(sum(bounce_list) / 1000)
    print("The average number of bounce: ", sum(bounce_list) / len(bounce_list))


if __name__ == "__main__":
    main()
