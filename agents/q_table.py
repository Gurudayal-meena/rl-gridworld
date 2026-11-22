import numpy as np
import random
import pickle
import os

class QTableAgent:
    
    def __init__(self, n_states, n_actions, lr=0.1, gamma=0.99, 
                 epsilon=1.0, eps_decay=0.995, eps_min=0.05):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        
        # Initialize Q-table with zeros
        self.q_table = np.zeros((n_states, n_actions))
        
        # Statistics tracking
        self.training_stats = {
            'episodes': 0,
            'total_steps': 0,
            'epsilon_history': [],
            'q_value_stats': []
        }

    def select_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        else:
            return int(np.argmax(self.q_table[state]))

    def update(self, state, action, reward, next_state, done):
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * np.max(self.q_table[next_state])
        
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.lr * td_error
        
        self.training_stats['total_steps'] += 1

    def end_episode(self):
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)
        self.training_stats['episodes'] += 1
        self.training_stats['epsilon_history'].append(self.epsilon)
        
        q_stats = {
            'mean': np.mean(self.q_table),
            'std': np.std(self.q_table),
            'max': np.max(self.q_table),
            'min': np.min(self.q_table)
        }
        self.training_stats['q_value_stats'].append(q_stats)

    def get_policy(self):
        return np.argmax(self.q_table, axis=1)

    def get_value_function(self):
        return np.max(self.q_table, axis=1)

    def save(self, filepath):
        save_data = {
            'q_table': self.q_table,
            'epsilon': self.epsilon,
            'training_stats': self.training_stats,
            'hyperparams': {
                'n_states': self.n_states,
                'n_actions': self.n_actions,
                'lr': self.lr,
                'gamma': self.gamma,
                'initial_epsilon': self.initial_epsilon,
                'eps_decay': self.eps_decay,
                'eps_min': self.eps_min
            }
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)

    def load(self, filepath):
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        self.q_table = save_data['q_table']
        self.epsilon = save_data['epsilon']
        self.training_stats = save_data['training_stats']
        
        # Load hyperparameters
        hyperparams = save_data['hyperparams']
        self.n_states = hyperparams['n_states']
        self.n_actions = hyperparams['n_actions']
        self.lr = hyperparams['lr']
        self.gamma = hyperparams['gamma']
        self.initial_epsilon = hyperparams['initial_epsilon']
        self.eps_decay = hyperparams['eps_decay']
        self.eps_min = hyperparams['eps_min']

    def reset_exploration(self):
        self.epsilon = self.initial_epsilon

    def get_action_probabilities(self, state):
        probs = np.full(self.n_actions, self.epsilon / self.n_actions)
        best_action = np.argmax(self.q_table[state])
        probs[best_action] += 1.0 - self.epsilon
        return probs

    def analyze_convergence(self):
        if len(self.training_stats['q_value_stats']) < 2:
            return None
        
        recent_stats = self.training_stats['q_value_stats'][-10:]
        early_stats = self.training_stats['q_value_stats'][:10]
        
        return {
            'q_value_change': abs(recent_stats[-1]['mean'] - recent_stats[0]['mean']),
            'early_vs_late_mean': recent_stats[-1]['mean'] - early_stats[-1]['mean'],
            'convergence_episodes': len(self.training_stats['q_value_stats']),
            'final_exploration': self.epsilon,
            'q_table_sparsity': np.mean(self.q_table == 0.0)
        }

    def __str__(self):
        return f"QTableAgent(states={self.n_states}, actions={self.n_actions}, epsilon={self.epsilon:.3f})"


class SarsaAgent(QTableAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "SARSA"

    def update(self, state, action, reward, next_state, next_action, done):
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * self.q_table[next_state, next_action]
        
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.lr * td_error
        
        self.training_stats['total_steps'] += 1


class ExpectedSarsaAgent(QTableAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "Expected-SARSA"

    def update(self, state, action, reward, next_state, done):
        if done:
            td_target = reward
        else:
            # Expected value over all actions weighted by policy probabilities
            action_probs = self.get_action_probabilities(next_state)
            expected_q = np.sum(action_probs * self.q_table[next_state])
            td_target = reward + self.gamma * expected_q
        
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.lr * td_error
        
        self.training_stats['total_steps'] += 1