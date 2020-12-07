import pandas as pd
import numpy as np


class Game:
    def __init__(self, game, s1, s2, length=1000):
        self.game = game
        self.s1 = s1.clone()
        self.s2 = s2.clone()
        self.length = length
        self.nb_cooperation_s1 = 0
        self.nb_cooperation_s2 = 0
        self.s1_score = 0
        self.s2_score = 0
        self.s1_rounds = []
        self.s2_rounds = []

    def reinit(self):
        self.s1_score = 0
        self.s2_score = 0
        self.s1_rounds = []
        self.s2_rounds = []

    def run(self):
        self.reinit()
        for tick in range(0, self.length):
            c1 = self.s1.getAction(tick).upper()
            c2 = self.s2.getAction(tick).upper()
            if c1 == "C":
                self.nb_cooperation_s1 += 1
            if c2 == "C":
                self.nb_cooperation_s2 += 1
            self.s1_rounds.append(c1)
            self.s2_rounds.append(c2)
            self.s1.update(c1, c2)
            self.s2.update(c2, c1)
            act = self.game.actions
            self.s1_score += self.game.scores["x"][act.index(c1), act.index(c2)]
            self.s2_score += self.game.scores["y"][act.index(c1), act.index(c2)]


class Tournament:
    def __init__(self, res_mat, strategies, length=1000):
        self.strategies = strategies
        self.res_mat = res_mat
        self.length = length
        size = len(strategies)
        df = pd.DataFrame(np.zeros((size, size + 1), dtype=np.int64))
        df.columns, df.index = (
            [s.name for s in self.strategies] + ["Total"],
            [s.name for s in self.strategies],
        )
        self.matrix = df
        df2 = pd.DataFrame(np.zeros((size, size + 1), dtype=np.int64))
        df2.columns, df2.index = (
            [s.name for s in self.strategies] + ["Total"],
            [s.name for s in self.strategies],
        )
        self.cooperations = df2
        self.ran = False

    def run(self):
        self.ran = True
        for i in range(0, len(self.strategies)):
            for j in range(i, len(self.strategies)):
                game = Game(
                    self.res_mat, self.strategies[i], self.strategies[j], self.length
                )
                game.run()
                self.matrix.at[
                    self.strategies[i].name, self.strategies[j].name
                ] = game.s1_score

                if i != j:
                    self.matrix.at[
                        self.strategies[j].name, self.strategies[i].name
                    ] = game.s2_score

                self.cooperations.at[
                    self.strategies[i].name, self.strategies[j].name
                ] = game.nb_cooperation_s1

                if i != j:
                    self.cooperations.at[
                        self.strategies[j].name, self.strategies[i].name
                    ] = game.nb_cooperation_s2

        self.matrix["Total"] = self.matrix.sum(axis=1)
        self.matrix.sort_values(by="Total", ascending=False, inplace=True)
        rows = list(self.matrix.index) + ["Total"]
        self.matrix = self.matrix.reindex(columns=rows)

        self.cooperations["Total"] = self.cooperations.sum(axis=1)
        self.cooperations.sort_values(by="Total", ascending=False, inplace=True)
        rows = list(self.cooperations.index) + ["Total"]
        self.cooperations = self.cooperations.reindex(columns=rows)

        
class TournamentVictory:
    def __init__(self, tournament):
        self.tournament = tournament
        self.matrix = tournament.matrix.iloc[:, :-1].copy()
        
    def run(self):
        for i in range(self.matrix.shape[0]):
            for j in range(i+1):
                self.matrix.iloc[i, j] = np.sign(self.tournament.matrix.iloc[i, j] - self.tournament.matrix.iloc[j, i])
                self.matrix.iloc[j, i] = -1 * self.matrix.iloc[i, j]

        self.matrix["Total"] = self.matrix.sum(axis=1)
        self.matrix.sort_values(by="Total", ascending=False, inplace=True)
        rows = list(self.matrix.index) + ["Total"]
        self.matrix = self.matrix.reindex(columns=rows)



