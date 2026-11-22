import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class GridWorld:
    
    def __init__(self, n_rows=5, n_cols=5, start=(0,0), goal=(4,4), traps=None, max_steps=100):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.start = start
        self.goal = goal
        self.traps = traps if traps else []
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        self.agent_pos = list(self.start)
        self.steps = 0
        return self._get_state()

    def _get_state(self):
        return self.agent_pos[0] * self.n_cols + self.agent_pos[1]

    def _state_to_pos(self, state):
        return (state // self.n_cols, state % self.n_cols)

    def _in_bounds(self, r, c):
        return 0 <= r < self.n_rows and 0 <= c < self.n_cols

    def step(self, action):
        # Actions: 0=up, 1=right, 2=down, 3=left
        dr = [-1, 0, 1, 0]
        dc = [0, 1, 0, -1]
        
        r, c = self.agent_pos
        nr, nc = r + dr[action], c + dc[action]
        
        # Only move if within bounds
        if self._in_bounds(nr, nc):
            self.agent_pos = [nr, nc]
        
        self.steps += 1

        # Calculate reward
        done = False
        reward = -0.01  # Small step penalty to encourage efficiency
        
        if tuple(self.agent_pos) == self.goal:
            reward = 1.0
            done = True
        elif tuple(self.agent_pos) in self.traps:
            reward = -1.0
            done = True
        elif self.steps >= self.max_steps:
            reward = -0.5  # Penalty for timeout
            done = True

        return self._get_state(), reward, done, {}

    @property
    def n_states(self):
        return self.n_rows * self.n_cols

    @property
    def n_actions(self):
        return 4

    def render(self, mode='text'):
        if mode == 'text':
            self._render_text()
        elif mode == 'matplotlib':
            return self._render_matplotlib()

    def _render_text(self):
        grid = [["" for _ in range(self.n_cols)] for _ in range(self.n_rows)]
        
        # Mark goal
        gr, gc = self.goal
        grid[gr][gc] = "G"
        
        # Mark start (if different from goal)
        sr, sc = self.start
        if (sr, sc) != (gr, gc):
            grid[sr][sc] = "S"
        
        # Mark traps
        for tr, tc in self.traps:
            grid[tr][tc] = "T"
        
        # Mark agent
        r, c = self.agent_pos
        grid[r][c] = "A"
        
        print("\n".join(" ".join(row) for row in grid))
        print(f"Steps: {self.steps}/{self.max_steps}")

    def _render_matplotlib(self):
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Draw grid
        for i in range(self.n_rows + 1):
            ax.axhline(y=i, color='black', linewidth=1)
        for j in range(self.n_cols + 1):
            ax.axvline(x=j, color='black', linewidth=1)
        
        # Color cells
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                # Goal cell (green)
                if (r, c) == self.goal:
                    rect = patches.Rectangle((c, self.n_rows-1-r), 1, 1, 
                                           linewidth=1, edgecolor='black', 
                                           facecolor='lightgreen', alpha=0.7)
                    ax.add_patch(rect)
                    ax.text(c+0.5, self.n_rows-1-r+0.5, 'G', 
                           ha='center', va='center', fontsize=20, fontweight='bold')
                
                # Trap cells (red)
                elif (r, c) in self.traps:
                    rect = patches.Rectangle((c, self.n_rows-1-r), 1, 1, 
                                           linewidth=1, edgecolor='black', 
                                           facecolor='lightcoral', alpha=0.7)
                    ax.add_patch(rect)
                    ax.text(c+0.5, self.n_rows-1-r+0.5, 'T', 
                           ha='center', va='center', fontsize=20, fontweight='bold')
                
                # Start cell (blue)
                elif (r, c) == self.start:
                    rect = patches.Rectangle((c, self.n_rows-1-r), 1, 1, 
                                           linewidth=1, edgecolor='black', 
                                           facecolor='lightblue', alpha=0.7)
                    ax.add_patch(rect)
                    ax.text(c+0.5, self.n_rows-1-r+0.5, 'S', 
                           ha='center', va='center', fontsize=20, fontweight='bold')
        
        # Agent position (yellow circle)
        r, c = self.agent_pos
        circle = patches.Circle((c+0.5, self.n_rows-1-r+0.5), 0.3, 
                               facecolor='yellow', edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(c+0.5, self.n_rows-1-r+0.5, 'A', 
               ha='center', va='center', fontsize=16, fontweight='bold')
        
        ax.set_xlim(0, self.n_cols)
        ax.set_ylim(0, self.n_rows)
        ax.set_aspect('equal')
        ax.set_title(f'GridWorld Environment - Steps: {self.steps}/{self.max_steps}', fontsize=16)
        ax.set_xticks(range(self.n_cols))
        ax.set_yticks(range(self.n_rows))
        
        return fig

    def get_optimal_policy(self):
        # Value iteration implementation
        V = np.zeros(self.n_states)
        policy = np.zeros(self.n_states, dtype=int)
        gamma = 0.99
        theta = 1e-6
        
        while True:
            delta = 0
            for s in range(self.n_states):
                v = V[s]
                r, c = self._state_to_pos(s)
                
                # Skip if terminal state
                if (r, c) == self.goal or (r, c) in self.traps:
                    continue
                
                action_values = []
                for a in range(self.n_actions):
                    # Simulate action
                    old_pos = self.agent_pos.copy()
                    old_steps = self.steps
                    self.agent_pos = [r, c]
                    self.steps = 0
                    
                    next_state, reward, done, _ = self.step(a)
                    if done:
                        action_values.append(reward)
                    else:
                        action_values.append(reward + gamma * V[next_state])
                    
                    # Restore state
                    self.agent_pos = old_pos
                    self.steps = old_steps
                
                V[s] = max(action_values)
                policy[s] = np.argmax(action_values)
                delta = max(delta, abs(v - V[s]))
            
            if delta < theta:
                break
        
        return policy, V

    def record_episode(self, agent, max_steps=None):
        if max_steps is None:
            max_steps = self.max_steps
        
        episode = []
        state = self.reset()
        total_reward = 0
        
        for step in range(max_steps):
            action = agent.select_action(state)
            episode.append({
                'step': step,
                'state': state,
                'position': tuple(self.agent_pos),
                'action': action,
                'action_name': ['up', 'right', 'down', 'left'][action]
            })
            
            next_state, reward, done, _ = self.step(action)
            total_reward += reward
            
            if done:
                episode.append({
                    'step': step + 1,
                    'state': next_state,
                    'position': tuple(self.agent_pos),
                    'action': None,
                    'action_name': 'terminal'
                })
                break
            
            state = next_state
        
        return episode, total_reward


class StochasticGridWorld(GridWorld):
    def __init__(self, success_prob=0.8, **kwargs):
        self.success_prob = success_prob
        super().__init__(**kwargs)
    
    def step(self, action):
        # With probability (1 - success_prob), take a random action instead
        if random.random() > self.success_prob:
            action = random.randint(0, 3)
        
        return super().step(action)


# Predefined environments for easy testing
def make_simple_grid():
    return GridWorld(
        n_rows=5, n_cols=5, 
        start=(0, 0), goal=(4, 4), 
        traps=[(2, 2)], 
        max_steps=50
    )

def make_maze_grid():
    return GridWorld(
        n_rows=7, n_cols=7,
        start=(0, 0), goal=(6, 6),
        traps=[(1, 1), (1, 2), (2, 1), (3, 3), (3, 4), (4, 3), (5, 5)],
        max_steps=100
    )