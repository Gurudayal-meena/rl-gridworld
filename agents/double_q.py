import numpy as np
import random
import pickle
import os

class DoubleQAgent:
    
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
        
        # Initialize two Q-tables with zeros
        self.q_table_1 = np.zeros((n_states, n_actions))
        self.q_table_2 = np.zeros((n_states, n_actions))
        
        # Statistics tracking
        self.training_stats = {
            'episodes': 0,
            'total_steps': 0,
            'epsilon_history': [],
            'q_value_stats': [],
            'q1_updates': 0,
            'q2_updates': 0
        }

    def select_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        else:
            # Use average of both Q-tables for action selection
            combined_q = self.q_table_1[state] + self.q_table_2[state]
            return int(np.argmax(combined_q))

    def update(self, state, action, reward, next_state, done):
        if random.random() < 0.5:
            # Update Q1 using Q2 for evaluation
            if done:
                td_target = reward
            else:
                # Use Q1 to select action, Q2 to evaluate
                best_next_action = np.argmax(self.q_table_1[next_state])
                td_target = reward + self.gamma * self.q_table_2[next_state, best_next_action]
            
            td_error = td_target - self.q_table_1[state, action]
            self.q_table_1[state, action] += self.lr * td_error
            self.training_stats['q1_updates'] += 1
        else:
            # Update Q2 using Q1 for evaluation
            if done:
                td_target = reward
            else:
                # Use Q2 to select action, Q1 to evaluate
                best_next_action = np.argmax(self.q_table_2[next_state])
                td_target = reward + self.gamma * self.q_table_1[next_state, best_next_action]
            
            td_error = td_target - self.q_table_2[state, action]
            self.q_table_2[state, action] += self.lr * td_error
            self.training_stats['q2_updates'] += 1
        
        self.training_stats['total_steps'] += 1

    def end_episode(self):
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)
        self.training_stats['episodes'] += 1
        self.training_stats['epsilon_history'].append(self.epsilon)
        
        # Record Q-value statistics for combined tables
        combined_q = self.q_table_1 + self.q_table_2
        q_stats = {
            'mean': np.mean(combined_q),
            'std': np.std(combined_q),
            'max': np.max(combined_q),
            'min': np.min(combined_q),
            'q1_mean': np.mean(self.q_table_1),
            'q2_mean': np.mean(self.q_table_2),
            'bias_estimate': np.mean(np.abs(self.q_table_1 - self.q_table_2))
        }
        self.training_stats['q_value_stats'].append(q_stats)

    def get_policy(self):
        combined_q = self.q_table_1 + self.q_table_2
        return np.argmax(combined_q, axis=1)

    def get_value_function(self):
        combined_q = self.q_table_1 + self.q_table_2
        return np.max(combined_q, axis=1)

    def get_q_tables(self):
        return self.q_table_1.copy(), self.q_table_2.copy()

    def get_bias_analysis(self):
        if len(self.training_stats['q_value_stats']) == 0:
            return None
        
        recent_stats = self.training_stats['q_value_stats'][-10:]
        
        analysis = {
            'q1_mean': np.mean([s['q1_mean'] for s in recent_stats]),
            'q2_mean': np.mean([s['q2_mean'] for s in recent_stats]),
            'combined_mean': np.mean([s['mean'] for s in recent_stats]),
            'bias_estimate': np.mean([s['bias_estimate'] for s in recent_stats]),
            'q1_updates_ratio': self.training_stats['q1_updates'] / max(1, self.training_stats['total_steps']),
            'q2_updates_ratio': self.training_stats['q2_updates'] / max(1, self.training_stats['total_steps']),
        }
        
        # Estimate overestimation bias compared to single Q-learning
        single_q_estimate = (self.q_table_1 + self.q_table_2) / 2
        max_single = np.max(single_q_estimate, axis=1)
        double_q_estimate = []
        
        for s in range(self.n_states):
            # Simulate double Q action selection
            if np.sum(self.q_table_1[s]) > np.sum(self.q_table_2[s]):
                best_action = np.argmax(self.q_table_1[s])
                value = self.q_table_2[s, best_action]
            else:
                best_action = np.argmax(self.q_table_2[s])
                value = self.q_table_1[s, best_action]
            double_q_estimate.append(value)
        
        double_q_estimate = np.array(double_q_estimate)
        analysis['overestimation_reduction'] = np.mean(max_single - double_q_estimate)
        
        return analysis

    def save(self, filepath):
        save_data = {
            'q_table_1': self.q_table_1,
            'q_table_2': self.q_table_2,
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
        
        self.q_table_1 = save_data['q_table_1']
        self.q_table_2 = save_data['q_table_2']
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
        combined_q = self.q_table_1[state] + self.q_table_2[state]
        best_action = np.argmax(combined_q)
        probs[best_action] += 1.0 - self.epsilon
        return probs

    def analyze_convergence(self):
        if len(self.training_stats['q_value_stats']) < 2:
            return None
        
        recent_stats = self.training_stats['q_value_stats'][-10:]
        early_stats = self.training_stats['q_value_stats'][:10]
        
        convergence_analysis = {
            'q_value_change': abs(recent_stats[-1]['mean'] - recent_stats[0]['mean']),
            'early_vs_late_mean': recent_stats[-1]['mean'] - early_stats[-1]['mean'],
            'convergence_episodes': len(self.training_stats['q_value_stats']),
            'final_exploration': self.epsilon,
            'q1_sparsity': np.mean(self.q_table_1 == 0.0),
            'q2_sparsity': np.mean(self.q_table_2 == 0.0),
            'table_similarity': np.corrcoef(self.q_table_1.flatten(), self.q_table_2.flatten())[0, 1]
        }
        
        return convergence_analysis

    def compare_with_single_q(self, single_q_table):
        combined_q = (self.q_table_1 + self.q_table_2) / 2
        
        comparison = {
            'value_difference': np.mean(np.abs(combined_q - single_q_table)),
            'overestimation_in_single': np.mean(single_q_table - combined_q),
            'policy_agreement': np.mean(np.argmax(combined_q, axis=1) == np.argmax(single_q_table, axis=1)),
            'max_value_difference': np.max(np.abs(combined_q - single_q_table)),
        }
        
        return comparison

    def __str__(self):
        return f"DoubleQAgent(states={self.n_states}, actions={self.n_actions}, epsilon={self.epsilon:.3f})"