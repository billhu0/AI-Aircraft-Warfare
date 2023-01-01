import numpy
import pandas
import random
import util
import state
import featureExtractor
import math


class QLearning:
    def __init__(self, learningRate=0.01, discount=0.9, explorationRate=0.3, numTraining=100): #调整了一下epsilon
        self.actions = ["right", "left", "up", "down", "bomb"]
        self.alpha = learningRate
        self.gamma = discount
        self.epsilon = explorationRate
        self.numTraining = numTraining
        self.qLearningTable = pandas.DataFrame(columns=self.actions, dtype=numpy.float64)
        self.values = util.Counter()

    def getQValue(self, state, action):
        if self.values[(state, action)] == None:
            return 0.0
        else:
            return self.values[(state, action)]

    def computeValueFromQValues(self, state):
        maxValue = float('-Inf')
        for action in self.actions:
            maxValue = max(maxValue, self.getQValue(state, action))
        return maxValue

    def computeActionFromQValues(self, state):
        maxValue = float('-Inf')
        bestAction = None
        for action in self.actions:
            if self.getQValue(state, action) > maxValue:
                maxValue = self.getQValue(state, action)
                bestAction = action
        return bestAction

    def getAction(self, state):
        prob = self.epsilon
        if util.flipCoin(prob):
            action = random.choice(self.actions)
        else:
            action = self.computeActionFromQValues(state)
        return action


class approximateQlearning(QLearning):
    def __init__(self):
        QLearning.__init__(self)
        self.weights = state.weight()
        self.features = util.Counter()

    def getQValue(self, state, action):
        qValue = 0
        self.features = featureExtractor.getFeatures(state, action)
        # print(self.weights)
        for i in self.features:
            qValue += self.features[i] * self.weights[i]
        return qValue

    # return action with highest Q-Value
    def computeActionFromQValues(self, state):
        maxValue = float('-Inf')
        bestAction = None
        for action in self.actions:
            value = self.getQValue(state, action)
            if value > maxValue:
                maxValue = value
                bestAction = action
        return bestAction

    def getAction(self, state):
        prob = self.epsilon
        if util.flipCoin(prob):
            action = random.choice(self.actions)
        else:
            action = self.computeActionFromQValues(state)
        return action

    def update(self, state, action, nextState, reward):
        difference = reward + self.gamma * \
            self.computeValueFromQValues(nextState) - self.getQValue(state, action)
        for feature in self.features:
            self.weights[feature] += self.alpha * difference * self.features[feature]
        self.weights.normalize()
        # print(self.weights)

    def computeValueFromQValues(self, state):
        maxValue = float('-Inf')
        for action in self.actions:
            maxValue = max(maxValue, self.getQValue(state, action))
        return maxValue
    def StoreWeights(self):
        with open('parameter.txt', 'w') as f:
            for key in self.weights.keys():
                if key == self.weights.keys()[-1]:
                    f.write("%f" % (self.weights[key]))
                else: 
                    f.write("%f," % (self.weights[key]))

