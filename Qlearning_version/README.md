# Playing Aircraft Warfare --- Approximate Q-Learning version

### Requirements 

- numpy
- pandas
- pygame

### Usage

To watch our Q-learning-based AI playing the game, use 

```python
python3 main.py 
```

To manually control the game, use 

```python
python3 main.py -m 
```

### Code stucture

The code structure is 

```
.
├── README.md               # ReadMe
├── bullet.py               # Game asset 
├── enemy.py                # Game asset
├── featureExtractor.py     # Feature extractor (calculate features from state)
├── font                    # Game assets
│   └── font.ttf
├── images                  # Game assets
│   └── ...
├── main.py                 # The entry of the program.
├── myplane.py              # Game asset
├── parameter.txt           # The stored parameters for the Q-learning.
├── q_learning.py           # The main class of Q-learning.
├── record.txt              # The highest score achieved in the game.
├── sound                   # Game assets
│   └── ...
├── state.py                # The state class used in Q-learning.
├── supply.py               # Game asset
├── util.py                 # Utility functions.
└── weights.txt             # The stored parameters for the Q-learning.
```


