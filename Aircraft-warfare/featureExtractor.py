import util
import pandas
import math
from typing import * 


""" 找最近的敌机 """
def find_the_nearest_plane(state, action):
    state.enemyPos.sort(key=lambda x: manhattanDistance(state.mePos, x))
    if len(state.enemyPos) < 3:
        return state.enemyPos
    return state.enemyPos[0:3]


def plane_move(pos1, move):
    if move == "up":
        return (pos1[0], pos1[1]-1)
    elif move == "down":
        return (pos1[0], pos1[1]+1)
    elif move == "left":
        return (pos1[0]-1, pos1[1])
    elif move == "right":
        return (pos1[0]+1, pos1[1])
    elif move == "bomb":
        return pos1
    else:
        return pos1


def calculate_distance_score(pos1, pos2, move):
    # 此函数计算我方飞机和其他飞机的距离分数，在小于某个阈值时，我方飞机偏向于远离其他飞机，大于某个阈值时，我方飞机偏向于靠近其他飞机（尽量采取左右移动）
    if pos2 == None:
        if manhattanDistance(plane_move(pos1, move), (200,500)) < manhattanDistance(pos1, (200,500)):
            # if move == "up" or move == "down":
            #     return 1500
            # else:
                
            return 1000
        return 100
    if 0.7*abs(pos1[1]-pos2[1])+0.3*abs(pos1[0]-pos2[0])< 70:
        if manhattanDistance(plane_move(pos1, move), pos2) > manhattanDistance(pos1, pos2):
            if move == "left" or  move == "right":
                return 3000
            else:
                return 1000
        else:
            return 300
    else:
        if move != "left" and move != "right":
            return 100
        else:
            if manhattanDistance(plane_move(pos1, move), pos2) < manhattanDistance(pos1, pos2):
                return 1000
            else:
                return 300


def calculate_get_gameprops(pos1, pos2, move):
    if pos2 == None:
        return 0
    if manhattanDistance(plane_move(pos1, move), pos2) < manhattanDistance(pos1, pos2):
        return 2000
    else:
        return 100


def manhattanDistance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
    if pos1 is None or pos2 is None:
        return 0
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def getFeatures(state, action):
    """
    这里就是approximate q_learning的feature——extratcor部分

    简单陈述一下要写的内容：
    """

    feature = util.Counter()
    gama = 0.1
    
    # TODO: 这一块暂时还无法运行, 需要改一改
    # feature1：计算我放飞机到最近的三个敌机之间的距离分数
    if action != "bomb":
        position = state.mePos
        bombs_pos = state.bomb_supply_pos
        double_bullet_pos = state.bullet_supply_pos
        enemyPosList = state.enemyPos.copy()
        if len(enemyPosList) > 0:
            enemyPosList.sort(key=lambda x: manhattanDistance(state.mePos, x))
        enemyPos_1 = enemyPosList[0] if len(enemyPosList) > 0 else None
        enemyPos_2 = enemyPosList[1] if len(enemyPosList) > 1 else None
        # enemyPos_3 = enemyPosList[2] if len(enemyPosList) > 2 else None # 这里的enemyPos_1, enemyPos_2, enemyPos_3是我方飞机到最近的三个敌机的距离
        feature["distance_score"] = calculate_distance_score(position, enemyPos_1, action) 
        # + gama*calculate_distance_score(
        #     position, enemyPos_2, action) 
            #+ gama*gama*calculate_distance_score(position, enemyPos_3, action)
        feature["get_bombs"] = calculate_get_gameprops(
            position, bombs_pos, action)
        feature["get_double_bullet"] = calculate_get_gameprops(
            position, double_bullet_pos, action)
        feature["bomb"] =300
    else:
        enemyPosList = state.enemyPos.copy()
        if len(enemyPosList) > 0:
            enemyPosList.sort(key=lambda x: manhattanDistance(state.mePos, x))
        enemyPos_1 = enemyPosList[0] if len(enemyPosList) > 0 else None
        if state.bomb_num > 0:
            if state.enemy_num > 6:
                    feature["bomb"] = 500*state.enemy_num
            else:
                feature["bomb"] = 400*state.enemy_num
            if enemyPos_1 !=None and manhattanDistance(state.mePos, enemyPos_1) < 30:
                feature["bomb"] = 6000
        else:
            feature["bomb"] = 100
        feature["bomb"]=50
        feature["distance_score"] = 50
        feature["get_bombs"] = 50
        feature["get_double_bullet"] = 50
    return feature
