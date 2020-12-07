import numpy as np
import pandas as pd
import math


class ResultMat:
    def __init__(self, scores, actions):
        self.actions = actions
        m = np.array(scores, dtype=[("x", object), ("y", object)])

        self.size = int(math.sqrt(len(scores)))
        self.scores = m.reshape(self.size, self.size)

    def print(self):
        game = pd.DataFrame(np.nan, self.actions, self.actions, dtype=object)
        for i in range(self.size):
            for j in range(self.size):
                game.iat[i, j] = self.scores[i][j]
        print(game)
