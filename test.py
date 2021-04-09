import Functions as f

# a list of strategies
class s():
    random = 0
    randomWinningPosition = 1
    minimax = 2
    abPruning = 3
    human = 4


# set up depth for minimax strategy
f.depth = 6

scores, winP, perRoundTime = f.Tournament(rounds= 2, ifPrint= True, xStrategy = s.minimax, oStrategy= s.human)

f.plot_hist(scores, '')
