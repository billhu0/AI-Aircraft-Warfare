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

# if gpu is to be used
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
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=3, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),            
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),            
        )
        self.layer1 = nn.Linear(18560, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x=self.conv_layer1(x)
        x=self.conv_layer2(x)
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

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
action_space = range(6)
n_actions = len(action_space)


plane = game.GameState()
action0 = torch.tensor([[0]],device=device, dtype=torch.long) 
observation0, reward0, terminal,_ = plane.frame_step(action0)

observation0 = cv2.cvtColor(cv2.resize(observation0, (80, 80)), cv2.COLOR_BGR2GRAY)
ret, observation0 = cv2.threshold(observation0,1,255,cv2.THRESH_BINARY)
observation0 = np.reshape(observation0,(80,80,1))
# observation0 = torch.flatten(torch.tensor(observation0))
n_observations = len(observation0)
# print(n_observations)
# print("------------------------")
# print(observation0)
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[np.random.choice(action_space)]], device=device, dtype=torch.long)


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
	# plane = game.GameState()
	# action0 = np.array([1,0,0])
	# observation0, reward0, terminal = plane.frame_step(action0)
	# observation0 = cv2.cvtColor(cv2.resize(observation0, (80, 80)), cv2.COLOR_BGR2GRAY)
	# ret, observation0 = cv2.threshold(observation0,1,255,cv2.THRESH_BINARY)
    if args.mode == 'display':
        target_net.load_state_dict(torch.load('./model/target.pkl'))
        policy_net.load_state_dict(torch.load('./model/policy.pkl'))
        target_net.to(device)
        policy_net.to(device)
        plane = game.GameState()
        action0 = torch.tensor([[0]],device=device, dtype=torch.long)  # [1,0,0]do nothing,[0,1,0]left,[0,0,1]right
        observation0, reward0, terminal,_ = plane.frame_step(action0)
        observation0,temp = preprocess(observation0)
        # print("-------------------------")
        print(temp.shape)
        # observation0 = torch.flatten(torch.tensor(observation0))
        state = torch.tensor(observation0, dtype=torch.float32, device=device).unsqueeze(0)
        for t in count():
            # print(t)
            action = select_action(state)
            #observation, reward, terminated, truncated, _ = env.step(action.item())
            observation,reward,terminated,score = plane.frame_step(action)
            # if reward > max_score:
            #     with open("score.txt","w") as f:
            #         f.write(str(score))
            observation,temp = preprocess(observation)
            # observation = torch.flatten(torch.tensor(observation))
            reward = torch.tensor([reward], device=device)
            #done = terminated or truncated
            done = terminated

            if terminated:
                next_state = None
                # with open("score.txt","a") as f:
                #     f.write(str(score))
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            # Move to the next state
            state = next_state

            if done:
                break
        return
    max_score = 0
    num_episodes = 600
    for i_episode in range(num_episodes):
        plane = game.GameState()
        action0 = torch.tensor([[0]],device=device, dtype=torch.long)  # [1,0,0]do nothing,[0,1,0]left,[0,0,1]right
        observation0, reward0, terminal,_ = plane.frame_step(action0)
        # cv2.imshow('imshow',observation0)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        observation0,temp = preprocess(observation0)
        print(temp.shape)
        
        # print("-------------------------")
        print("Episode {} Start.".format(i_episode))
        # observation0 = torch.flatten(torch.tensor(observation0))
        state = torch.tensor(observation0, dtype=torch.float32, device=device).unsqueeze(0)

        for t in count():
            # print(t)
            action = select_action(state)
            #observation, reward, terminated, truncated, _ = env.step(action.item())
            observation,reward,terminated,score = plane.frame_step(action)
            # if reward > max_score:
            #     with open("score.txt","w") as f:
            #         f.write(str(score))
            observation,temp = preprocess(observation)
            # cv2.imshow('imshow',cv2.resize(np.reshape(temp,(480,700)),(350,240)))
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # observation = torch.flatten(torch.tensor(observation))
            reward = torch.tensor([reward], device=device)
            #done = terminated or truncated
            done = terminated

            if terminated:
                next_state = None
                if score >= max_score:
                    max_score = score
                    print("Max Score Update: {}.".format(max_score))
                    torch.save(target_net.state_dict(), './model/target_best.pth')
                    torch.save(policy_net.state_dict(), './model/policy_best.pth')
                with open("score.txt","a") as f:
                    f.write("Score: {}, Max Score: {}, Episode: {}.".format(score,max_score,i_episode))
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_durations.append(t + 1)
                break
        if i_episode % 10 == 0:
            print("We have finsh: "+str(i_episode))
            torch.save(target_net_state_dict,"./model/target_{}.pth".format(int(i_episode/10)))
            torch.save(policy_net_state_dict,"./model/policy_{}.pth".format(int(i_episode/10)))
            
    torch.save(target_net_state_dict,"./model/target.pth")
    torch.save(policy_net_state_dict,"./model/policy.pth")

def main():
	playPlane()

if __name__ == '__main__':
	main()