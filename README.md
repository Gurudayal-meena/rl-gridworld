
# Reinforcement Learning GridWorld Project

**Course:** CS 329 - Reinforcement Learning  
**Institute:** Indian Institute of Technology Gandhinagar  
**Project Title:** A Comparative Study of Temporal-Difference Reinforcement Learning Algorithms in GridWorld

## Group Members

| Name | Student ID | Assigned Contribution |
|---|---|---|
| Gurudayal Meena | 23110125 | GridWorld environment implementation, configuration, reward design, transition logic, and testing |
| Ramavath Chanti | 23110270 | Q-Learning, SARSA, and training setup |
| Mayank Gulati | 22110146 | Expected SARSA, experiments, learning curves, and comparative analysis |
| R. Ganesh | 23110271 | Double Q-Learning, evaluation, policy visualization, and integration |

The final project includes the technical components corresponding to these assignments. Individual task fulfilment is reported in the final project report and should be confirmed by the group.

---

## 1. Project Overview

This project implements and compares four temporal-difference reinforcement learning algorithms in two GridWorld environments.

The objective is to study how different reinforcement learning algorithms learn optimal navigation policies, avoid traps, and reach goal states.

The algorithms implemented are:

1. **Q-Learning:** An off-policy temporal-difference learning algorithm.
2. **SARSA:** An on-policy temporal-difference learning algorithm.
3. **Expected SARSA:** Uses the expected next-state action value for learning.
4. **Double Q-Learning:** Maintains two Q-tables to reduce overestimation bias.

The algorithms are evaluated on:

- **Simple GridWorld (5x5):** A small environment containing one trap.
- **Maze GridWorld (7x7):** A larger environment containing multiple traps.

Both environments use deterministic transitions and a single learning agent.

---

## 2. Project Objectives

The main objectives are:

- Implement four tabular reinforcement learning algorithms.
- Develop two GridWorld environments with different difficulty levels.
- Compare learning performance during training.
- Evaluate the final greedy policies.
- Study success rates, cumulative rewards, and episode lengths.
- Visualize learning results and environment configurations.
- Analyze differences between algorithms and environments.

---

## 3. Project Structure

```text
rl-gridworld/
|
|-- agents/
|   |-- q_table.py
|   |-- double_q.py
|
|-- environment/
|   |-- gridworld.py
|
|-- train.py
|-- eval.py
|-- analysis.py
|-- requirements.txt
|
|-- results/
|   |-- *.json
|   |-- *_agent.pkl
|   |-- *_plot.png
|
|-- docs/
|   |-- figures/
|       |-- gridworld_layouts.png
|       |-- training_success.png
|       |-- environment_gap.png
|       |-- greedy_reward.png
|
|-- README.md
```

### Description of Important Files

- `environment/gridworld.py` - Implements the GridWorld environments, actions, rewards, transitions, and terminal conditions.
- `agents/q_table.py` - Contains Q-Learning, SARSA, and Expected SARSA implementations.
- `agents/double_q.py` - Implements Double Q-Learning.
- `train.py` - Trains the agents and stores training results.
- `eval.py` - Evaluates trained agents and visualizes learned policies.
- `analysis.py` - Analyzes and compares experimental results.
- `requirements.txt` - Lists the Python dependencies.
- `results/` - Stores the original experiment outputs.
- `docs/figures/` - Contains figures used in the final project report.

---

## 4. Installation and Setup

### Requirements

- Python 3
- NumPy
- Matplotlib
- Pandas
- Seaborn
- SciPy

### Installation

Clone or download this GitHub repository.

Open a terminal in the project folder and run:

```bash
python -m pip install -r requirements.txt
```

This installs the dependencies required by the project.

---

## 5. GridWorld Environments

### 5.1 Simple GridWorld (5x5)

The Simple GridWorld contains 25 states.

**Environment configuration:**

- Grid size: 5x5
- Starting position: (0, 0)
- Goal position: (4, 4)
- Trap position: (2, 2)
- Maximum episode length: 50 steps

The objective is to reach the goal while avoiding the trap.

### 5.2 Maze GridWorld (7x7)

The Maze GridWorld contains 49 states and multiple traps.

**Environment configuration:**

- Grid size: 7x7
- Starting position: (0, 0)
- Goal position: (6, 6)
- Maximum episode length: 100 steps

**Trap positions:**

```text
(1,1), (1,2), (2,1),
(3,3), (3,4), (4,3), (5,5)
```

The maze presents a more challenging navigation problem because of its larger state space and additional trap locations.

### 5.3 Action Space

The agent can perform four actions:

```text
0 - Up
1 - Right
2 - Down
3 - Left
```

### 5.4 Reward Structure

| Condition | Reward |
|---|---:|
| Reaching the goal | +1.00 |
| Entering a trap | -1.00 |
| Normal movement step | -0.01 |
| Reaching the maximum step limit | -0.50 |

The agent learns to reach the goal while minimizing unnecessary steps and avoiding traps.

---

## 6. Reinforcement Learning Algorithms

### 6.1 Q-Learning

Q-Learning is an off-policy temporal-difference control algorithm.

It updates action values using the maximum estimated value of the next state.

### 6.2 SARSA

SARSA is an on-policy temporal-difference control algorithm.

It updates action values using the next action selected by the current policy.

### 6.3 Expected SARSA

Expected SARSA calculates the expected next-state action value under the epsilon-greedy policy.

It uses the expectation over possible next actions rather than one sampled next action.

### 6.4 Double Q-Learning

Double Q-Learning uses two separate Q-tables.

One table selects the best next action, while the other evaluates that action.

This helps reduce overestimation bias.

---

## 7. Experimental Configuration

The original archived experiments use the following parameters:

| Parameter | Value |
|---|---|
| Training episodes | 1000 |
| Learning rate (alpha) | 0.1 |
| Discount factor (gamma) | 0.99 |
| Initial epsilon | 1.0 |
| Epsilon decay | 0.995 |
| Minimum epsilon | 0.05 |
| Random seed | 42 |
| Greedy evaluation episodes | 100 |

The same training configuration was used for the four algorithms in both environments.

There are eight archived training runs in total:

```text
4 algorithms x 2 environments = 8 experiments
```

---

## 8. How to Run the Project

### 8.1 Train Q-Learning

For the Simple GridWorld:

```bash
python train.py --agent qtable --env standard --episodes 1000
```

For the Maze GridWorld:

```bash
python train.py --agent qtable --env maze --episodes 1000
```

### 8.2 Train SARSA

```bash
python train.py --agent sarsa --env standard --episodes 1000
```

```bash
python train.py --agent sarsa --env maze --episodes 1000
```

### 8.3 Train Expected SARSA

```bash
python train.py --agent expected_sarsa --env standard --episodes 1000
```

```bash
python train.py --agent expected_sarsa --env maze --episodes 1000
```

### 8.4 Train Double Q-Learning

```bash
python train.py --agent double_q --env standard --episodes 1000
```

```bash
python train.py --agent double_q --env maze --episodes 1000
```

**Important:** The training script uses `standard` for the 5x5 GridWorld, while the saved results use filenames containing `simple`.

---

## 9. Evaluation

The trained agents can be evaluated using `eval.py`.

To visualize a learned policy in the Simple GridWorld:

```bash
python eval.py --agent-path "PATH_TO_SAVED_AGENT.pkl" --env simple --policy
```

To inspect Q-values in the Maze GridWorld:

```bash
python eval.py --agent-path "PATH_TO_SAVED_AGENT.pkl" --env maze --qtable
```

Replace `PATH_TO_SAVED_AGENT.pkl` with the path to an actual `.pkl` file inside the `results/` directory.

**Note:** The evaluation script uses `simple`, not `standard`, for the 5x5 environment.

---

## 10. Result Analysis

Run the analysis script:

```bash
python analysis.py
```

To save analysis plots:

```bash
python analysis.py --save-plots
```

The project stores experiment outputs in the `results/` directory.

These include:

- JSON files containing training metrics.
- Pickle files containing trained agents.
- PNG files containing learning curves.
- Evaluation results recorded with the experiments.

---

## 11. Experimental Results

### 11.1 Training Success Rates

The following results were recorded during the original 1000-episode training runs.

| Algorithm | Simple GridWorld | Maze GridWorld |
|---|---:|---:|
| Q-Learning | 87.3% | 67.0% |
| SARSA | 92.2% | 76.4% |
| Expected SARSA | 92.8% | 76.7% |
| Double Q-Learning | 93.1% | 70.6% |

The original implementation calculates training success using an episode-reward threshold greater than 0.5.

These results are based on one archived random seed and should not be interpreted as averages across multiple independent runs.

### 11.2 Results Discussion

In the Simple GridWorld, Double Q-Learning achieved the highest recorded training success rate of 93.1%.

Expected SARSA and SARSA also performed well, with success rates of 92.8% and 92.2%, respectively.

In the Maze GridWorld, Expected SARSA achieved the highest recorded training success rate of 76.7%, followed by SARSA at 76.4%.

Q-Learning achieved lower training success rates in both environments.

All four algorithms experienced a reduction in training success when moving from the Simple GridWorld to the more challenging Maze GridWorld.

### 11.3 Greedy Evaluation Results

After training, the agents were evaluated using greedy policies with exploration disabled.

| Algorithm | Simple Success | Maze Success |
|---|---:|---:|
| Q-Learning | 100% | 100% |
| SARSA | 100% | 100% |
| Expected SARSA | 100% | 100% |
| Double Q-Learning | 100% | 100% |

The archived evaluation results also reported:

- Mean return for Simple GridWorld: **0.93**
- Mean return for Maze GridWorld: **0.89**

Each evaluation used 100 episodes.

Because the environment is deterministic and starts from the same position, repeated greedy evaluations can produce identical outcomes.

Training success and greedy evaluation success are different metrics and should not be confused.

---

## 12. Screenshots and Visualizations

The following visualizations summarize the original GridWorld configurations and saved experiment results.

### 12.1 GridWorld Environment Layouts

![GridWorld Environment Layouts](<img width="1705" height="848" alt="gridworld_layouts" src="https://github.com/user-attachments/assets/aaca3e6d-8415-4fa8-b96c-d84892c16dec" />
)

The figure shows the Simple GridWorld and Maze GridWorld environments, including the starting positions, goal positions, and traps.

### 12.2 Training Success Rate Comparison

![Training Success Comparison](<img width="1780" height="752" alt="training_success" src="https://github.com/user-attachments/assets/1ce7eee1-3f84-44ed-aa59-63325906fdce" />
)

The graph compares the recorded training success rates of the four algorithms across both environments.

### 12.3 Performance Difference Between Environments

![Environment Performance Gap](<img width="1726" height="680" alt="environment_gap" src="https://github.com/user-attachments/assets/e3ebbb83-e7d8-4dcd-914f-117aee12dc25" />
)

This figure shows the differences in training success rates between the Simple GridWorld and Maze GridWorld.

### 12.4 Greedy Evaluation Rewards

![Greedy Evaluation Rewards](<img width="1691" height="680" alt="greedy_reward" src="https://github.com/user-attachments/assets/fc0d0de8-c709-4fc8-8c8a-368da1992aae" />
)

This graph presents the mean episode rewards obtained during greedy evaluation.

### 12.5 Original Learning Curves

The `results/` directory contains the original learning curve plots generated by the project's training code.

These files use names ending in:

```text
_plot.png
```

The corresponding saved agent parameters use names ending in:

```text
_agent.pkl
```

The original JSON results are also preserved for verification.

---

## 13. Reproducibility

The final report is based on the original archived experiment results.

To verify the reported values:

1. Inspect the JSON result files inside `results/`.
2. Check the training parameters and recorded success metrics.
3. Compare the results with the tables in this README.
4. Inspect the original learning curves.
5. Evaluate the stored trained agents.
6. Rerun the training scripts if additional experiments are required.

The quantitative comparison plots in this README are derived from the saved experimental summaries.

The environment layout figure is based on the original environment configurations.

The project does not claim statistical significance from multiple random seeds.

---

## 14. Contribution

The project responsibilities were divided among four group members.

**Gurudayal Meena:** GridWorld environment design and implementation, state and action representations, reward structure, transitions, terminal conditions, environment testing, and navigation validation.

**Ramavath Chanti:** Q-Learning and SARSA implementation, training configuration, epsilon-greedy exploration, and comparison of learning behaviour.

**Mayank Gulati:** Expected SARSA implementation, experimental evaluation, training curves, success-rate plots, and comparative analysis.

**R. Ganesh:** Double Q-Learning implementation, evaluation pipeline, policy visualization, reward analysis, and algorithm comparison.

The repository contains the technical components corresponding to the four assignments. Individual completion statements are provided in the final report and should be confirmed by the group.

All members also participated in debugging, experimental discussions, comparative analysis, and report preparation.

---

## 15. Conclusion

This project compares four tabular temporal-difference reinforcement learning algorithms in two GridWorld environments.

The archived training experiments show that the algorithms perform differently across the Simple GridWorld and Maze GridWorld.

The more challenging Maze GridWorld produces lower training success rates, while the saved greedy evaluations show successful navigation for all four algorithms.

The comparison demonstrates the importance of distinguishing training performance from the performance of a final learned policy.

The project provides a practical understanding of Q-Learning, SARSA, Expected SARSA, and Double Q-Learning through environment implementation, experimentation, and performance analysis.

---

## References

1. Sutton, R. S., and Barto, A. G. (2018). *Reinforcement Learning: An Introduction*. MIT Press.
2. Watkins, C. J. C. H., and Dayan, P. (1992). Q-learning. *Machine Learning*.
3. van Seijen, H., et al. (2009). A Theoretical and Empirical Analysis of Expected Sarsa.
4. van Hasselt, H. (2010). Double Q-learning. *Advances in Neural Information Processing Systems*.
