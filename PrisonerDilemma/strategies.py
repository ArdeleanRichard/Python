import random


class Strategy:
    def setMemory(self, mem):
        pass

    def getAction(self, tick):
        pass

    def __copy__(self):
        pass

    def update(self, x, y):
        pass


class Cooperator(Strategy):
    def __init__(self):
        super().__init__()
        self.name = "cooperator"
        self.past = ""

    def getAction(self, tick):
        return "C"

    def clone(self):
        return Cooperator()

    def update(self, x, y):
        pass


class Defector(Strategy):
    def __init__(self):
        super().__init__()
        self.name = "defector"
        self.past = ""

    def getAction(self, tick):
        return "D"

    def clone(self):
        return Defector()

    def update(self, x, y):
        pass


class Randomer(Strategy):
    def __init__(self):
        super().__init__()
        self.name = "randomer"
        self.past = ""

    def getAction(self, tick):
        return "C" if random.uniform(0, 1) < 0.5 else "D"

    def clone(self):
        return Randomer()

    def update(self, x, y):
        pass


class Tft(Strategy):
    def __init__(self):
        super().__init__()
        self.name = "tft"
        self.past = ""

    def getAction(self, tick):
        return "C" if (tick == 0) else self.past[-1]

    def clone(self):
        return Tft()

    def update(self, my, his):
        self.past += his

