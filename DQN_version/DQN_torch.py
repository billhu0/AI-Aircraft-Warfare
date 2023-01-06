import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
import random
from itertools import count
from collections import deque,namedtuple
import cv2
import sys
import plane as game
import argparse


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, help="train/display")
args = parser.parse_args()

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        # self.conv_layer1 = nn.Sequential(
        #     nn.Conv2d(1, 16, kernel_size=5, stride=3, padding=2),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),            
        # )
        # self.conv_layer2 = nn.Sequential(
        #     nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),            
        # )
        # self.layer1 = nn.Linear(18560, 128)
        self.layer1 = nn.Linear(n_observations, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128,64)
        self.layer4 = nn.Linear(64, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        # x=self.conv_layer1(x)
        # x=self.conv_layer2(x)
        # x = x.reshape(x.size(0), -1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        # print(x.grad)
        return self.layer4(x)

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the AdamW optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# Get number of actions from gym action space
action_space = torch.arange(4)
n_actions = len(action_space)


plane = game.GameState()
action0 = torch.tensor([[0]],device=device, dtype=torch.long) 
observation0, reward0, terminal,_ = plane.frame_step(action0)


n_observations = len(observation0)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)


steps_done = 0


def select_action(plane_game, state):
    # s_cnt = 0
    # m_cnt = 0
    # b_cnt = 0
    # for enemy_1 in plane_game.small_enemies.sprites():
    #     if enemy_1.rect.top > 0:
    #         s_cnt += 1
    # for enemy_2 in plane_game.mid_enemies.sprites():
    #     if enemy_2.rect.top > 0:
    #         m_cnt += 1
    # for enemy_3 in plane_game.big_enemies.sprites():
    #     if enemy_3.rect.top > 0:
    #         b_cnt += 1
    # if s_cnt + 3*m_cnt + 5*b_cnt > 6:
    #     if plane_game.bomb_num:
    #         return torch.tensor([[3]], device=device, dtype=torch.long)
    # global steps_done
    # sample = random.random()
    # eps_threshold = EPS_END + (EPS_START - EPS_END) * \
    #     math.exp(-1. * steps_done / EPS_DECAY)
    # steps_done += 1
    # if sample > eps_threshold:
    #     with torch.no_grad():
    #         return policy_net(state).max(1)[1].view(1, 1)
    # else:
    #     return torch.tensor([[np.random.choice(action_space)]], device=device, dtype=torch.long)
    with torch.no_grad():
             return policy_net(state).max(1)[1].view(1, 1)


def cal_action(plane_game, state):
    with torch.no_grad():
        s_cnt = 0
        m_cnt = 0
        b_cnt = 0
        for enemy_1 in plane_game.small_enemies.sprites():
            if enemy_1.rect.top > 0 and enemy_1.active:
                s_cnt += 1
        for enemy_2 in plane_game.mid_enemies.sprites():
            if enemy_2.rect.top > 0 and enemy_2.active:
                m_cnt += 1
        for enemy_3 in plane_game.big_enemies.sprites():
            if enemy_3.rect.top > 0 and enemy_3.active:
                b_cnt += 1
        if s_cnt + 3*m_cnt + 5*b_cnt > 9:
            if plane_game.bomb_num:
                return torch.tensor([[3]], device=device, dtype=torch.long)
        if s_cnt + m_cnt + b_cnt == 0:
            return torch.tensor([[0]], device=device, dtype=torch.long)
        # print(state)
        state = state.reshape(-1)
        # print(state)
        me_center = np.array(state[0:2])
        delta_close = np.array(state[2:4])
        delta_bomb = np.array(state[4:6])
        delta_bullet = np.array(state[6:8])
        bomb_cnt = np.array(state[-3])
        is_double = np.array(state[-2])
        delta_supply = np.full(2,1e3)
        type_close = int(state[-1])
        dis_weight = np.array([0.3,0.7])
        dis_threshold = [0,70,100,140]
        # print(delta_close,type_close)
        if bomb_cnt < 3:
            delta_supply = delta_bomb
        if is_double == 0:
            if np.sum(dis_weight*np.abs(delta_supply)) > np.sum(dis_weight*np.abs(delta_bullet)):
                delta_supply = delta_bullet

        if np.sum(dis_weight*np.abs(delta_supply)) < np.sum(dis_weight*np.abs(delta_close)):
            if delta_supply[0] == 0:
                return torch.tensor([[0]], device=device, dtype=torch.long)
            elif delta_supply[0] < 0:
                return torch.tensor([[1]], device=device, dtype=torch.long)
            else :
                return torch.tensor([[2]], device=device, dtype=torch.long)
        if abs(delta_close[0]) <= 13/2 and is_double and type_close == 1:
            if me_center[0] >= 240:
                return torch.tensor([[1]], device=device, dtype=torch.long)
            else:
                return torch.tensor([[2]], device=device, dtype=torch.long)
        if np.sum(dis_weight*np.abs(delta_close)) < dis_threshold[type_close]:
            if delta_close[0] < 0:
                return torch.tensor([[2]], device=device, dtype=torch.long)
            else :
                return torch.tensor([[1]], device=device, dtype=torch.long)
        else :
            if delta_close[0] == 0:
                # if type_close == 1 and is_double:
                #     return torch.tensor([[1]], device=device, dtype=torch.long)
                return torch.tensor([[0]], device=device, dtype=torch.long)

            if delta_close[0] < 0:
                return torch.tensor([[1]], device=device, dtype=torch.long)
            else :
                return torch.tensor([[2]], device=device, dtype=torch.long)
        if delta_close[0] == 0 and type_close == 1 and is_double:
                return torch.tensor([[1]], device=device, dtype=torch.long)
episode_durations = []

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
   
    batch = Transition(*zip(*transitions))

   
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    
    state_action_values = policy_net(state_batch).gather(1, action_batch)

   
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


 # preprocess raw image to 80*80 gray image
def preprocess(img_obs):
	# img_obs = cv2.cvtColor(cv2.resize(img_obs, (80, 80)), cv2.COLOR_BGR2GRAY)#
	img_obs = cv2.cvtColor(img_obs, cv2.COLOR_BGR2GRAY)#
	ret, img_obs = cv2.threshold(img_obs,1,255,cv2.THRESH_BINARY)
	return np.reshape(img_obs,(1,480,700)),img_obs


def playPlane():
    if args.mode == 'display':
        
        plane = game.GameState()
        action0 = torch.tensor([[0]],device=device, dtype=torch.long)  
        observation0, reward0, terminal,_ = plane.frame_step(action0)
    
        state = torch.tensor(observation0, dtype=torch.float32, device=device).unsqueeze(0)
        for t in count():
            action = cal_action(plane,state)
            
            observation,reward,terminated,score = plane.frame_step(action)
            
            reward = torch.tensor([reward], device=device)
           
            done = terminated

            if terminated:
                next_state = None
               
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            state = next_state

            if done:
                break
        return
    max_score = 0
    #num_episodes = 600
    # for i_episode in range(num_episodes):
    i_episode = 0
    while i_episode <= 5:
        plane = game.GameState()
        action0 = torch.tensor([[0]],device=device, dtype=torch.long)  
        observation0, reward0, terminal,_ = plane.frame_step(action0)
        
        
        # print("-------------------------")
        print("Episode {} Start.".format(i_episode))
        
        state = torch.tensor(observation0, dtype=torch.float32, device=device).unsqueeze(0)

        for t in count():
            # print(t)
            action = select_action(plane, state)
            action_score = policy_net(state).reshape(-1)
            ideal_action = cal_action(plane,state)
            
            observation,reward,terminated,score = plane.frame_step(action)
            
            reward = torch.tensor([reward], device=device)
            done = terminated

            if terminated:
                next_state = None
                if score >= max_score:
                    max_score = score
                    print("Max Score Update: {}.".format(max_score))
                    torch.save(target_net.state_dict(), './model/target_best.pth')
                    torch.save(policy_net.state_dict(), './model/policy_best.pth')
                with open("score.txt","a") as f:
                    f.write("Score: {}, Max Score: {}, Episode: {}.\n".format(score,max_score,i_episode))
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            state = next_state

            
            c_loss = nn.CrossEntropyLoss()

            loss = c_loss(action_score,\
                torch.tensor(F.one_hot(ideal_action[0][0],num_classes=4),dtype=float,device=device,requires_grad=True))
            
            optimizer.zero_grad()
        
            loss.backward()
            
            torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
            optimizer.step()

            if done:
                episode_durations.append(t + 1)
                break
        if i_episode % 5 <= 4:
            print("We have finsh: "+str(i_episode))
            
        i_episode += 1    

def main():
	playPlane()

if __name__ == '__main__':
	main()