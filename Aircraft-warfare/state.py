import numpy
"""
这一部分仿照佘以宁他们写的,具体实现可以后续讨论
"""
class state:
    def __init__(self, mePos, score=0, enemy_num=0, enemyPos = [], bullet_supply_pos = None, bomb_supply_pos = None, life_num=3, is_double_bullet=False, bomb_num=3):
        """
        先把定义写出,state怎么定义可以后续讨论

        score: 分数
        
        enemy_num: 敌机数量
        enemyPos: 敌机位置, list: [(left, top), ...], if no enemy: list = []

        bullet_supply_pos: 子弹补给位置, tuple: (left, top), if no supple: None
        bomb_supply_pos: 炸点补给位置, tuple: (left, top), if no supple: None

        mePos: 我方位置, tuple: (left, top)
        bomb_num: 炸弹个数
        is_double_bullet: 子弹是否为强化状态, bool
        life_num: 生命值
        """
        self.score = score

        self.enemy_num = enemy_num
        self.enemyPos = enemyPos

        self.bullet_supply_pos = bullet_supply_pos
        self.bomb_supply_pos = bomb_supply_pos

        self.mePos = mePos
        self.bomb_num = bomb_num
        self.is_double_bullet = is_double_bullet
        self.life_num = life_num
        
    def assignData(self, score, mePos, enemy_num, enemyPos, bullet_supply_pos, bomb_supply_pos, life_num, is_double_bullet, bomb_num):
        """
        observation是一个dict(也可能是counter),总之observation用来更新
        """
        self.score = score

        self.enemy_num = enemy_num
        self.enemyPos = enemyPos

        self.bullet_supply_pos = bullet_supply_pos
        self.bomb_supply_pos = bomb_supply_pos

        self.mePos = mePos
        self.bomb_num = bomb_num
        self.is_double_bullet = is_double_bullet
        self.life_num = life_num


class weight(dict):
    def __init__(self):
        """
        这里weight应该是feature的参数
        """
        dict.__init__(self,{'distance_score':1,'get_bomb':1,"get_double_bullet" :1,"bomb" :1})
        
    def normalize(self):
        sumOfValues = sum([v for v in self.values()])
        for i in self.keys():
            self[i] /= sumOfValues