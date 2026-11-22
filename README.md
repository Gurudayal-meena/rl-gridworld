# Reinforcement Learning GridWorld Project

**CS 329 - Reinforcement Learning**  
**IIT Gandhinagar**

**Group Members:**
- Gurudayal Meena (23110125)
- Ramavath Chanti (23110270)
- Mayank Gulati   (22110146)
- R.Ganesh        (23110271)

## Overview

This project implements and compares different reinforcement learning algorithms on GridWorld environments:

- Q-Learning: Off-policy learning
- SARSA: On-policy learning  
- Expected SARSA: Expected value approach
- Double Q-Learning: Reduces overestimation bias

## Project Structure

```
rl-gridworld/
├── environment/
│   └── gridworld.py      # Environment implementation
├── agents/
│   ├── q_table.py        # Q-Learning and SARSA agents
│   └── double_q.py       # Double Q-Learning agent
├── train.py             # Training script
├── eval.py              # Evaluation script
├── analysis.py          # Analysis and comparison
├── results/             # Training results
└── README.md
```

## Setup

```bash
pip install numpy matplotlib pandas seaborn scipy
```

## Usage

### Training Agents

```bash
# Train Q-Learning on standard environment
python train.py --agent qtable --env standard --episodes 1000

# Train SARSA on maze environment  
python train.py --agent sarsa --env maze --episodes 1500
```

### Available Options

**Agents:** qtable, sarsa, expected_sarsa, double_q
**Environments:** standard, maze

### Evaluation

```bash
# Visualize trained agent policy
python eval.py --agent-path results/qtable_standard_*.pkl --env standard --policy

# Analyze Q-table values
python eval.py --agent-path results/sarsa_maze_*.pkl --env maze --qtable
```

### Analysis

```bash
# Run complete analysis
python analysis.py

# Save plots to files
python analysis.py --save-plots
```

## Environments

### Standard Grid (5x5)
Basic environment with one trap at (2,2) and goal at (4,4).

### Maze Grid (7x7) 
Complex environment with multiple obstacles requiring navigation.

## Results

Results are saved in the `results/` directory:
- `.json` files contain training metrics
- `.pkl` files contain trained agent parameters
- `.png` files contain learning curves and visualizations

## Implementation Details

All algorithms use epsilon-greedy exploration with learning rate 0.1 and discount factor 0.99.