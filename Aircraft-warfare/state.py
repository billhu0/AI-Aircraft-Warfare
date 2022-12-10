import numpy
"""
这一部分仿照佘以宁他们写的,具体实现可以后续讨论
"""
class state:
    def __init__(self):
        """
        先把定义写出,state怎么定义可以后续讨论
        """
        pass
    def assignData(self,observation):
        """
        observation是一个dict(也可能是counter),总之observation用来更新
        """
        pass


class weight(dict):
    def __init__(self):
        """
        这里weight应该是feature的参数
        """
        pass
    def normalize(self):
        sumOfValues = sum([v for v in self.values()])
        for i in self.keys():
            self[i] /= sumOfValues