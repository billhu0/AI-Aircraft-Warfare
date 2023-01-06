import random


def flipCoin(prob: float) -> bool:
    return random.random() < prob


class Counter(dict):
    def __getitem__(self, key):
        self.setdefault(key, 0)
        return dict.__getitem__(self, key)
