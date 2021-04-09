from collections import namedtuple
import numpy as np
import time
import matplotlib.pyplot as plt
import random

# python dictionary to map integers (1, -1, 0) to characters ('x', 'o', ' ')
symbols = {1: 'x',
           -1: 'o',
           0: ' '}

# set minimax depth 0
depth = 0

nodeDict = {}  # dictionary of node identifiers
nodeSuccDict = {}  # dictionary of node successors
nodePredDict = {}  # dictionary of node predecessors
nodeTransDict = {} # a dictionary of transpositions
nodeTransTypeDict = {} # a dictionary of transpositions
nodeUtilDict = {}  # create utility dictionary for xx
nodeMinMaxDict = {}  # minimax dictionary
hashDict = {}  # hash dictionary


# check if move still possible
def move_still_possible(S):
    # if there's 0 in game state
    return not (S[S == 0].size == 0)


# winning move - if there's 3 in one line
def move_was_winning_move(S, p):
    if np.max((np.sum(S, axis=0)) * p) == 3:
        return True
    if np.max((np.sum(S, axis=1)) * p) == 3:
        return True
    if (np.sum(np.diag(S)) * p) == 3:
        return True
    if (np.sum(np.diag(np.rot90(S))) * p) == 3:
        return True
    return False


def print_game_state(S):
    B = np.copy(S).astype(object)
    for n in [-1, 0, 1]:
        B[B == n] = symbols[n]
    print(B)


# return available position's coordinate
def available_cords(S):
    xs, ys = np.where(S == 0)

    # ture 1D to 2D cords
    cords = list(zip(xs, ys))
    return cords


# player moves according to strategy
def move_according_to_strategy(S, p, strategy):
    if strategy == 0:
        return move_at_random(S, p)
    if strategy == 1:
        return move_according_winning_positions(S, p)
    if strategy == 2:
        return minmax(S, p, p)
    if strategy == 3:
        return ab_pruning(S, p)
    if strategy == 4:
        return human_player(S, p)


# random
def move_at_random(S, p):
    availableCords = available_cords(S)
    randomCords = random.choice(availableCords)
    S[randomCords] = p
    return S


# move according to the best available winning position map
def move_according_winning_positions(S, p):
    # check where is empty on board
    availableCords = available_cords(S)

    # positions are ranked into 3 tiers
    winningTierCords = [
                   [(0, 0), (0, 2), (2, 0), (2, 2)],
                   [(0, 1), (1, 0), (1, 2), (2, 1)],
                   [(1, 1)]
    ]

    # start from the lowest tier
    for cord in winningTierCords:
        # get the intersection of current tier and empty boards
        intersection = set(cord) & set(availableCords)

        # if intersects, randomly choose from the intersection
        if intersection:
            randomCords = random.choice(list(intersection))
    S[randomCords] = p
    return S


# get a unique hash number of the current game state
def get_hash_number(S):
    # reshape the 2d array to 1d, make o = 0, empty = 1, and x = 2
    SPlusOne = np.array(S).reshape(9) + 1

    # reset hashNumber
    hashNumber = 0

    for i in range(9):
        # since there are 3 states: 0, 1, 2, we can consider each game state as a nine-digit base-3 numeral
        # then return the corresponding integer as the hash value
        # eg: [2, 0, 0, 0, 0, 0, 0, 1, 0] => 2*3^8 + 1*3^1
        hashNumber += SPlusOne[i] * (3 ** (8-i))
    return hashNumber


# check if the current game state is a transposition of a reference state
def check_transposition(S, reference):
    for i in range(4):
        # original game state is flipped upside down
        SUpSideDown = np.flipud(S)

        # both S and SUpSideDown are rotated 0°, 90°, 180°, 270° then compared with the reference, return transposition index i
        if np.array_equal(np.rot90(S, i), reference):
            return i
        if np.array_equal(np.rot90(SUpSideDown, i), reference):
            return i + 4


# do transposition according to index i
def do_transposition(S, i):
    # original game state is flipped upside down
    SUpSideDown = np.flipud(S)

    # rotate both S and SUpSideDown 0°, 90°, 180°, 270° according to the index
    if i < 4:
        return np.rot90(S, i)
    else:
        return np.rot90(SUpSideDown, i - 4)


# evaluate a game state. maxPlayer is the player using the strategy
# game states are evaluated in maxPlayer's perspective
def evaluate_game_state(S, maxPlayer, p):
    # if player wins, return p * infinity
    if move_was_winning_move(S, p):
        # a relative large number 10 + how many empty space on board when there's a win
        # so winning faster has higher scores
        return (10 + len(np.where(S == 0)[0])) * p * maxPlayer

    # else, counting the winning lines
    else:
        T1 = np.copy(S)

        # make all empty position to p
        T1[T1 == 0] = p
        n1 = num_winning_lines(T1, p)

        T2 = np.copy(S)
        # make all empty position to -p
        T2[T2 == 0] = -p
        n2 = num_winning_lines(T2, -p)

        # get the evaluation
        return (n1 - n2) * p * maxPlayer


# count total winning lines
def num_winning_lines(S, p):
    cs = np.sum(S, axis=0) * p
    rs = np.sum(S, axis=1) * p
    s1 = cs[cs == 3].size
    s2 = rs[rs == 3].size
    s3 = 0
    if np.sum(np.diag(S)) * p == 3:
        s3 = 1
    s4 = 0
    if np.sum(np.diag(np.rot90(S))) * p == 3:
        s4 = 1
    return s1 + s2 + s3 + s4


def buildTree(S, maxPlayer, p, node, depth):
    succ = []   # a list to hold all the possible successor nodes
    # if depth is 0, we evaluate the game and assign it to utility dict
    if depth == 0:
        nodeUtilDict[node] = evaluate_game_state(S, maxPlayer, p)
        return

    # transpose the current state, check if exists in hashDict
    # 8 ways of transpositions
    for i in range(8):
        STrans = do_transposition(S, i)

        # get the hash number of the transposed state, check if it exists in hash dict
        # if it does, it means we had the same state before
        if hashDict.get(get_hash_number(STrans)) is not None:

            # assign current game state to the new node
            nodeDict[node] = S

            # assign current transposition's 'prototype' node and record the transposition type
            nodeTransDict[node] = hashDict.get(get_hash_number(STrans))
            nodeTransTypeDict[node] = i

            # assign it's successors same as the existed one
            nodeSuccDict[node] = nodeSuccDict.get(hashDict.get(get_hash_number(STrans)))
            return

    # if not in hash dict, assign new item
    hashDict[get_hash_number(S)] = node

    # if there's a winning move
    if move_was_winning_move(S, p):
        # evaluate the winning move
        nodeUtilDict[node] = evaluate_game_state(S, maxPlayer, p)
        return

    # if game ends in a draw
    elif S[S == 0].size == 0:
        nodeUtilDict[node] = 0
        return

    # if S is not terminal: switch player & compute successors
    else:
        # switch player
        p *= -1

        # get empty position cords
        rs, cs = np.where(S == 0)

        # loop all the possible successors of the current state
        for j in range(rs.size):
            # player p moves
            Ssucc = np.copy(S)
            Ssucc[rs[j], cs[j]] = p

            # get the max key of node dictionary, make a new key
            newnode = max(nodeDict.keys()) + 1

            # assign the current successor to node dictionary
            nodeDict[newnode] = Ssucc

            # append current successor node to succ
            succ.append(newnode)

            # assign current successor node's parent node to nodePredDict
            nodePredDict[newnode] = node

        # assign all the successors to nodeSuccDict
        nodeSuccDict[node] = succ

        # continue recursively
        for s in succ:
            buildTree(nodeDict[s], maxPlayer, p, s, depth-1)


# minimax functions
def maxNodeUtil(node):
    # check if node is already in minimax dict
    if node in nodeMinMaxDict:
        return nodeMinMaxDict[node]

    # check in trans dict, if node has other transpositions in minimax dict
    if nodeTransDict.get(node) in nodeMinMaxDict:
        nodeMinMaxDict[node] = nodeMinMaxDict.get(nodeTransDict.get(node))
        return nodeMinMaxDict[node]

    # check if node in utility dict
    if node in nodeUtilDict:
        nodeMinMaxDict[node] = nodeUtilDict[node]
        return nodeMinMaxDict[node]

    # set mmv to - infinite
    mmv = -np.inf

    # look into it's successors
    # some nodes are pruned, so they don't have child, we try first
    try:
        for s in nodeSuccDict[node]:
            mmv = max(mmv, minNodeUtil(s))
        nodeMinMaxDict[node] = mmv
        return mmv
    # if the node is pruned, we set the value to - inf
    except:
        nodeMinMaxDict[node] = -np.inf
        return -np.inf


# same as above
def minNodeUtil(node):
    if node in nodeMinMaxDict:
        return nodeMinMaxDict[node]
    if nodeTransDict.get(node) in nodeMinMaxDict:
        nodeMinMaxDict[node] = nodeMinMaxDict.get(nodeTransDict.get(node))
        return nodeMinMaxDict[node]
    if node in nodeUtilDict:
        nodeMinMaxDict[node] = nodeUtilDict[node]
        return nodeMinMaxDict[node]
    mmv = np.inf
    try:
        for s in nodeSuccDict[node]:
            mmv = min(mmv, maxNodeUtil(s))
        nodeMinMaxDict[node] = mmv
        return mmv
    except:
        nodeMinMaxDict[node] = np.inf
        return np.inf


# get the best move according to game tree
def best_move(node):
    # a list of best moves
    bestMoves = []

    # set initial moveValue to - inf
    moveValue = -np.inf

    # look into node's successors, find the move with max value
    for i in nodeSuccDict.get(node):
        # print()
        # print_game_state(nodeDict.get(i))
        # print(i, nodeMinMaxDict.get(i))

        # find the max value in the successors
        if moveValue <= nodeMinMaxDict.get(i):
            moveValue = nodeMinMaxDict.get(i)

    # do the loop again, find all the moves that has the same max value
    for i in nodeSuccDict.get(node):
        if moveValue == nodeMinMaxDict.get(i):
            # put them all in the list
            bestMoves.append(nodeDict.get(i))
    # choose a random one
    bestMove = random.choice(bestMoves)
    # print_game_state(bestMove)
    # print()
    return bestMove


# minmax strategy
def minmax(S, maxPlayer, p):
    # empty all the dicts
    nodeDict.clear()
    nodeSuccDict.clear()
    nodeUtilDict.clear()
    nodeTransDict.clear()
    nodeTransTypeDict.clear()
    nodeMinMaxDict.clear()
    hashDict.clear()

    node = 0
    nodeDict[node] = S

    # build game tree
    buildTree(S, maxPlayer, -p, node, depth)

    print('minimax ', len(nodeDict))

    # assign minmax value
    maxNodeUtil(node)

    # check if current state is a transposition of an existed state
    i = nodeTransTypeDict.get(node)
    # if is copy the existed one and reverse the transposing
    if i:
        S = do_transposition(nodeTransDict.get(node), i)
    # otherwise get the best move
    else:
        S = best_move(node)
    return S


# αβ pruning
# build the game tree one layer down
def build_tree_one_level(S, p, node):
    succ = []

    # transpose the current state, check if exists in hashDict
    # 8 ways of transpositions
    for i in range(8):
        STrans = do_transposition(S, i)

        # get the hash number of the transposed state, check if it exists in hash dict
        # if it does, it means we had the same state before
        if hashDict.get(get_hash_number(STrans)) is not None:

            # assign current game state to the new node
            nodeDict[node] = S

            # assign current transposition's 'prototype' node and record the transposition type
            nodeTransDict[node] = hashDict.get(get_hash_number(STrans))
            nodeTransTypeDict[node] = i

            # assign it's successors same as the existed one
            nodeSuccDict[node] = nodeSuccDict.get(hashDict.get(get_hash_number(STrans)))
            return

    # if not in hash dict, assign new item
    hashDict[get_hash_number(S)] = node

    rs, cs = np.where(S == 0)
    for j in range(rs.size):
        Ssucc = np.copy(S)
        Ssucc[rs[j], cs[j]] = p

        # get the max key of node dictionary, make a new key
        newnode = max(nodeDict.keys()) + 1

        # assign the current successor to node dictionary
        nodeDict[newnode] = Ssucc

        # append current successor node to succ
        succ.append(newnode)

        # assign current successor node's parent node to nodePredDict
        nodePredDict[newnode] = node

    nodeSuccDict[node] = succ


# ab pruning get max value
def max_val(n, maxPlayer, p, a, b):
    # get game sate S
    S = nodeDict.get(n)

    # check if win or still possible to move
    if move_was_winning_move(S, p):
        nodeUtilDict[n] = evaluate_game_state(S, maxPlayer, p)
        return nodeUtilDict[n]
    elif not move_still_possible(nodeDict.get(n)):
        nodeUtilDict[n] = 0
        return nodeUtilDict[n]

    # switch player
    p = -p

    # build the tree one level down
    build_tree_one_level(nodeDict.get(n), p, n)

    # get max value down to the path
    for i in nodeSuccDict[n]:
        a = max(a, min_val(i, maxPlayer, p, a, b))
        if a >= b:
            return a
    return a


# same as above
def min_val(n, maxPlayer, p, a, b):
    S = nodeDict.get(n)
    if move_was_winning_move(S, p):
        nodeUtilDict[n] = evaluate_game_state(S, maxPlayer, p)
        return nodeUtilDict[n]
    elif not move_still_possible(nodeDict.get(n)):
        nodeUtilDict[n] = 0
        return nodeUtilDict[n]
    p = -p
    build_tree_one_level(nodeDict.get(n), p, n)
    for i in nodeSuccDict[n]:
        b = min(b, max_val(i, maxPlayer, p, a, b))
        if b <= a:
            return b
    return b


# ab_pruning strategy
def ab_pruning(S, p):
    # empty all dictionary
    nodeDict.clear()
    nodeSuccDict.clear()
    nodeUtilDict.clear()
    nodeTransDict.clear()
    nodeTransTypeDict.clear()
    nodeMinMaxDict.clear()
    hashDict.clear()

    node = 0
    nodeDict[0] = S

    # start pruning
    max_val(node, p, -p, -np.inf, np.inf)

    # get minimax value
    maxNodeUtil(node)

    # used to check the tree size, please ignore
    print('abpruning ', len(nodeDict))

    # get the best move
    S = best_move(node)

    return S


# a human player, input 1-9
def human_player(S, p, wasError = False):
    if np.array_equal(S, np.zeros((3, 3), dtype=int)):
        print_game_state(S)
    if not wasError:
        move = input("Please press 1-9 to play.")
    else: move = input()
    try:
        move = int(move)
        moveY = (move - 1) // 3
        moveX = (move - 1) % 3
        if (moveY, moveX) in available_cords(S):
            S[moveY, moveX] = p
            return S
        else:
            print("Position is not available! Try again!")
            return human_player(S, p, True)
    except ValueError:
        print("Input is not 1-9! Try again!")
        return human_player(S, p, True)


def play(ifPrint = True, xStrategy = 0, oStrategy = 0):
    # initialize an empty tic tac toe board
    gameState = np.zeros((3, 3), dtype=int)

    # initialize the player who moves first (either +1 or -1)
    player = 1

    # initialize a move counter
    mvcntr = 1

    # initialize a flag that indicates whether or not the game has ended
    noWinnerYet = True

    while move_still_possible(gameState) and noWinnerYet:
        # turn current player number into player symbol
        name = symbols[player]
        if ifPrint: print('%s moves' % name)

        if player == 1:
            strategy = xStrategy
        else:
            strategy = oStrategy

        # let current player move at random
        gameState = move_according_to_strategy(gameState, player, strategy)

        # print current game state
        if ifPrint:
            print_game_state(gameState)

        # evaluate current game state
        if move_was_winning_move(gameState, player):
            if ifPrint: print('player %s wins after %d moves' % (name, mvcntr))
            noWinnerYet = False
            return player, gameState

        # switch current player and increase move counter
        player *= -1
        mvcntr += 1

    if noWinnerYet:
        if ifPrint: print('game ended in a draw')
        return 0, gameState


# tournament according to rounds, strategies
def Tournament(rounds = 1000, ifPrint = False, xStrategy = 0, oStrategy = 0):
    # set start time
    start_time = time.time()

    # score counter [x , draw, o]
    scores = [0, 0, 0]

    # winning position counter
    winPositions = np.zeros((3, 3), dtype=int)

    # start the tournament
    for i in range(rounds):
        # record the score and winning position each round
        score, winPosition = play(ifPrint, xStrategy, oStrategy)

        # score is either 1, 0 or -1, so '2-(score + 1)' is either 0, 1, 2, which is correspond to scores' index
        scores[2-(score + 1)] += 1

        # record current wp
        winPositions += np.where(winPosition == score, 1, 0)

    # get total run time and per round time
    runTime = round((time.time() - start_time), 4)
    perRoundTime = round(runTime/rounds, 4)
    print(f'-Total Run Time: {runTime} -Per Round Time: {perRoundTime}')

    # normalize winning position
    winPosistionsNormalized = np.round(np.array(winPositions) / np.sum(winPositions), 3)
    return scores, winPosistionsNormalized, perRoundTime


# for plotting score
def plot_hist(score_counter, title = ''):
    objects = ('X', 'Draws', 'O')
    y_pos = np.arange(len(objects))
    for i in range(len(objects)):
        plt.text(i, np.sum(score_counter) / 30, score_counter[i], ha="center", va="center")
    plt.bar(y_pos, score_counter, align='center', alpha=0.5, color='green')
    plt.xticks(y_pos, objects)
    plt.title(str(title), pad=15)
    plt.show()


# for plotting winning position
def plot_heat(wpcntr, title = ''):
    fig, ax = plt.subplots()
    plt.imshow(wpcntr, cmap='summer')

    ax.set_xticks(np.arange(len(wpcntr[0])))
    ax.set_yticks(np.arange(len(wpcntr)))
    ax.set_xticklabels(np.arange(len(wpcntr[0]))+1)
    ax.set_yticklabels(np.arange(len(wpcntr))+1)

    for i in range(len(wpcntr)):
        for j in range(len(wpcntr[0])):
            text = ax.text(j, i, wpcntr[i, j],
                           ha="center", va="center", color="Black")

    ax.set_title(str(title), pad=15)
    fig.tight_layout()
    plt.show()


# for plotting run time
def plot_list(result, title = ''):
    objects = []
    for i in range(len(result)):
        objects.append(i+1)
    y_pos = np.arange(len(objects))
    for i in range(len(objects)):
        plt.text(i, result[i] + np.sum(result) / (20 * len(objects)), result[i], ha="center", va="center")
    plt.bar(y_pos, result, align='center', alpha=0.5, color='green')
    plt.xticks(y_pos, objects)
    plt.title(str(title), pad=15)
    plt.show()



# a, b = Tournament(100, True, 0, 0)
# plot_hist(a, 1)
# plot_heat(b, 2)

# def pa(i):
#     print('node: ', i)
#     print_game_state(nodeDict.get(i))
#     print('succ: ', nodeSuccDict.get(i))
#     print('pred: ', nodePredDict.get(i))
#     print('trans: ', nodeTransDict.get(i))
#     print('type: ', nodeTransTypeDict.get(i))
#     print('util: ', nodeUtilDict.get(i))
#     print('mmv: ', nodeMinMaxDict.get(i))
#     # print(hashDict)
#     print()
