from ipd import *
from strategies import *
from resultmat import ResultMat

scores = [(3, 3), (0, 5), (5, 0), (1, 1)]


rm = ResultMat(scores, ["C", "D"])
print("---------Result Matrix----------")
rm.print()
players = [Cooperator(), Defector(), Randomer(), Tft()]
t = Tournament(rm, players)        # default: length=1000
t.run()
print("---------Tournament----------")
print(t.matrix)
tv = TournamentVictory(t)
tv.run()
print("---------Tournament Winner----------")
print(tv.matrix)
