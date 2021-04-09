import Functions as f


# a list of strategies
class s():
    random = 0
    randomWinningPosition = 1
    minimax = 2
    abPruning = 3
    human = 4


# # both players play random, get the scores and winning position
scores, winP, perRoundTime = f.Tournament(rounds= 10000, ifPrint= False, xStrategy = s.random, oStrategy= s.random)
f.plot_hist(scores, 'Random vs Random')
f.plot_heat(winP, 'Winning Position Heat Map')


# # x plays according to the winning position tier while o play at random
scores, winP, perRoundTime = f.Tournament(rounds= 10000, ifPrint= False, xStrategy = s.randomWinningPosition, oStrategy= s.random)
f.plot_hist(scores, 'Random Winning Position vs Random')


# x uses minimax while o plays according to the winning position
# set minimax depth to 3
f.depth = 3
scores, winP, perRoundTime = f.Tournament(rounds= 100, ifPrint= False, xStrategy = s.minimax, oStrategy= s.random)
f.plot_hist(scores, 'Minimax vs Random')


# both players use minimax
# set minimax depth to 6
f.depth = 6
scores, winP, perRoundTime = f.Tournament(rounds=100, ifPrint=False, xStrategy=s.minimax, oStrategy=s.minimax)
f.plot_hist(scores, 'Minimax vs Minimax')


# x uses minimax while o plays at random, set depth from 1 - 9
# list to record time
perRoundTimes = []
for i in range(9):
    f.depth = i + 1
    score, winP, perRoundTime = f.Tournament(rounds= 100, ifPrint= False, xStrategy = s.minimax, oStrategy= s.random)
    perRoundTimes.append(perRoundTime)

# plot per round time according to the depth
f.plot_list(perRoundTimes, 'Per Round Time(s) according to Minimax Depth')


# x uses ab pruning while o plays at random
score, winP, perRoundTime = f.Tournament(rounds= 100, ifPrint= False, xStrategy = s.abPruning, oStrategy= s.random)
f.plot_hist(score, 'AB Pruning vs Random')
