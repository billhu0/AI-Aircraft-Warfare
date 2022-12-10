import random

def flipCoin(prob):
    tmp = random.random()
    return tmp<prob

class Counter(dict):
    def __getitem__(self, key):
       self.setdefault(key,0)
       return dict.__getitem__(self,key)
