# CS181 Project --- Playing Aircraft Warfare Game with Reinforcement Learning

### Requirements

- numpy
- pandas
- pygame
- pytorch
- python-opencv

You can install these packages by the following command.

```sh
python3 -m pip install -r ./requirements.txt
```

### Usage

In our project, we implement an AI-agent for playing the Aircraft Warfare with Appriximate Q-learning method and Deep Q-learning method. The difficulty of the game can be adjusted by adjusting the speed. They are placed in `./Qlearning_version/` and `./DQN_Version/` respectively.

You can run the Q-learning version by the following command

```sh
cd Qlearning_version
python3 main.py
```

Or run the DQN version by the following command

```sh
cd DQN_version
python3 DQN_torch.py
```

### External resources

External resources we used:

- [https://github.com/yangshangqi/The-Python-code-implements-aircraft-warfare.git](https://github.com/yangshangqi/The-Python-code-implements-aircraft-warfare.git)
