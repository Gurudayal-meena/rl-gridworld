import argparse
import os
import numpy as np
import random
import time
from datetime import datetime
import json
import matplotlib.pyplot as plt

from environment.gridworld import GridWorld, make_simple_grid, make_maze_grid
from agents.q_table import QTableAgent, SarsaAgent, ExpectedSarsaAgent
from agents.double_q import DoubleQAgent

def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)

def create_environment(env_name, **kwargs):
    env_creators = {
        'standard': make_simple_grid,
        'maze': make_maze_grid,
        'custom': lambda: GridWorld(**kwargs)
    }
    
    if env_name not in env_creators:
        raise ValueError(f"Unknown environment: {env_name}")
    
    return env_creators[env_name]()

def create_agent(agent_type, n_states, n_actions, agent_config):
    agent_creators = {
        'qtable': QTableAgent,
        'sarsa': SarsaAgent,
        'expected_sarsa': ExpectedSarsaAgent,
        'double_q': DoubleQAgent
    }
    
    if agent_type not in agent_creators:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    return agent_creators[agent_type](n_states, n_actions, **agent_config)

def train_tabular_agent(env, agent, episodes, verbose=True):
    rewards = []
    episode_lengths = []
    success_count = 0
    
    start_time = time.time()
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        episode_length = 0
        done = False
        
        # For SARSA, we need the next action
        if isinstance(agent, SarsaAgent):
            action = agent.select_action(state)
        
        while not done:
            if not isinstance(agent, SarsaAgent):
                action = agent.select_action(state)
            
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            episode_length += 1
            
            if isinstance(agent, SarsaAgent):
                if not done:
                    next_action = agent.select_action(next_state)
                    agent.update(state, action, reward, next_state, next_action, done)
                    action = next_action
                else:
                    agent.update(state, action, reward, next_state, None, done)
            else:
                agent.update(state, action, reward, next_state, done)
            
            state = next_state
        
        # Episode finished
        agent.end_episode()
        rewards.append(total_reward)
        episode_lengths.append(episode_length)
        
        # Check if episode was successful (reached goal)
        if total_reward > 0.5:  # Assuming goal reward is 1.0
            success_count += 1
        
        # Print progress
        if verbose and (episode + 1) % max(1, episodes // 20) == 0:
            recent_rewards = rewards[-100:] if len(rewards) >= 100 else rewards
            success_rate = success_count / (episode + 1)
            elapsed_time = time.time() - start_time
            
            print(f"Episode {episode + 1}/{episodes} | "
                  f"Avg Reward: {np.mean(recent_rewards):.3f} | "
                  f"Success Rate: {success_rate:.2f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Time: {elapsed_time:.1f}s")
    
    return {
        'rewards': rewards,
        'episode_lengths': episode_lengths,
        'success_rate': success_count / episodes,
        'final_epsilon': agent.epsilon,
        'training_time': time.time() - start_time
    }

def evaluate_agent(env, agent, n_episodes=100, max_steps=None):
    if max_steps is None:
        max_steps = env.max_steps
    
    rewards = []
    episode_lengths = []
    success_count = 0
    
    for _ in range(n_episodes):
        state = env.reset()
        total_reward = 0
        episode_length = 0
        done = False
        
        while not done and episode_length < max_steps:
            action = agent.select_action(state, training=False)  # No exploration
            state, reward, done, _ = env.step(action)
            total_reward += reward
            episode_length += 1
        
        rewards.append(total_reward)
        episode_lengths.append(episode_length)
        
        if total_reward > 0.5:
            success_count += 1
    
    return {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'mean_length': np.mean(episode_lengths),
        'success_rate': success_count / n_episodes,
        'rewards': rewards
    }

def plot_training_results(results, title, save_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16)
    
    # Rewards over episodes
    axes[0, 0].plot(results['rewards'], alpha=0.3, color='blue', label='Episode Reward')
    
    # Smooth rewards with moving average
    window = max(1, len(results['rewards']) // 50)
    if len(results['rewards']) >= window:
        smooth_rewards = np.convolve(results['rewards'], 
                                   np.ones(window)/window, mode='valid')
        axes[0, 0].plot(range(len(smooth_rewards)), smooth_rewards, 
                       color='red', linewidth=2, label=f'Moving Avg ({window})')
    
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title('Learning Curve')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Episode lengths
    axes[0, 1].plot(results['episode_lengths'], alpha=0.6, color='green')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Episode Length')
    axes[0, 1].set_title('Episode Lengths Over Time')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Success rate (rolling window)
    window_size = max(1, len(results['rewards']) // 20)
    success_rates = []
    for i in range(len(results['rewards'])):
        start_idx = max(0, i - window_size)
        recent_rewards = results['rewards'][start_idx:i+1]
        success_rate = sum(1 for r in recent_rewards if r > 0.5) / len(recent_rewards)
        success_rates.append(success_rate)
    
    axes[1, 0].plot(success_rates, color='purple', linewidth=2)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Success Rate')
    axes[1, 0].set_title(f'Success Rate (Rolling Window: {window_size})')
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Loss tracking (if available)
    if 'losses' in results and results['losses']:
        # Smooth losses
        window = max(1, len(results['losses']) // 100)
        if len(results['losses']) >= window:
            smooth_losses = np.convolve(results['losses'], 
                                      np.ones(window)/window, mode='valid')
            axes[1, 1].plot(smooth_losses, color='orange', linewidth=2)
            axes[1, 1].set_xlabel('Training Step')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].set_title('Training Loss')
            axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'No loss data\n(Tabular Agent)', 
                       ha='center', va='center', transform=axes[1, 1].transAxes,
                       fontsize=12)
        axes[1, 1].set_title('Training Loss')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    return fig

def save_results(results, config, save_path):
    save_data = {
        'config': config,
        'results': results,
        'timestamp': datetime.now().isoformat()
    }
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    
    print(f"Results saved to {save_path}")

def run_experiment(config):
    print(f"Starting experiment with configuration:")
    print(f"Agent: {config['agent_type']}")
    print(f"Environment: {config['env_name']}")
    print(f"Episodes: {config['episodes']}")
    print(f"Seed: {config['seed']}")
    print("-" * 50)
    
    # Set random seed
    set_random_seeds(config['seed'])
    
    # Create environment
    env = create_environment(config['env_name'], **config.get('env_config', {}))
    print(f"Environment: {env.n_rows}x{env.n_cols} grid, {env.n_states} states, {env.n_actions} actions")
    
    # Create agent
    agent = create_agent(config['agent_type'], env.n_states, env.n_actions, 
                        config.get('agent_config', {}))
    print(f"Agent: {agent}")
    
    # Train agent
    print(f"\nTraining for {config['episodes']} episodes...")
    if config['agent_type'] in ['qtable', 'sarsa', 'expected_sarsa', 'double_q']:
        results = train_tabular_agent(env, agent, config['episodes'], 
                                    config.get('verbose', True))
    else:
        raise ValueError(f"Unknown agent type: {config['agent_type']}")
    
    print(f"\nTraining completed in {results['training_time']:.1f} seconds")
    print(f"Final success rate: {results['success_rate']:.3f}")
    print(f"Final exploration rate: {results['final_epsilon']:.3f}")
    
    # Evaluate agent
    if config.get('evaluate', True):
        print("\nEvaluating trained agent...")
        eval_results = evaluate_agent(env, agent, 
                                    n_episodes=config.get('eval_episodes', 100))
        results['evaluation'] = eval_results
        
        print(f"Evaluation results:")
        print(f"  Mean reward: {eval_results['mean_reward']:.3f} ± {eval_results['std_reward']:.3f}")
        print(f"  Mean episode length: {eval_results['mean_length']:.1f}")
        print(f"  Success rate: {eval_results['success_rate']:.3f}")
    
    # Save results
    if config.get('save_results', True):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = f"results/{config['agent_type']}_{config['env_name']}_{timestamp}.json"
        save_results(results, config, results_path)
        
        # Save agent
        if config.get('save_agent', True):
            agent_path = f"results/{config['agent_type']}_{config['env_name']}_{timestamp}_agent.pkl"
            agent.save(agent_path)
            print(f"Agent saved to {agent_path}")
    
    # Plot results
    if config.get('plot_results', True):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = f"results/{config['agent_type']}_{config['env_name']}_{timestamp}_plot.png"
        title = f"{config['agent_type'].upper()} on {config['env_name']} Environment"
        plot_training_results(results, title, plot_path)
    
    return results, agent

def main():
    parser = argparse.ArgumentParser(description='Train RL agents on GridWorld')
    parser.add_argument('--agent', type=str, choices=['qtable', 'sarsa', 'expected_sarsa', 'double_q'], 
                       default='qtable', help='Agent type')
    parser.add_argument('--env', type=str, choices=['standard', 'maze'], 
                       default='standard', help='Environment to use')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--epsilon', type=float, default=1.0, help='Initial exploration rate')
    parser.add_argument('--eps-decay', type=float, default=0.995, help='Exploration decay rate')
    parser.add_argument('--no-eval', action='store_true', help='Skip evaluation')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting')
    parser.add_argument('--no-save', action='store_true', help='Skip saving results')
    parser.add_argument('--quiet', action='store_true', help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    # Build configuration
    config = {
        'agent_type': args.agent,
        'env_name': args.env,
        'episodes': args.episodes,
        'seed': args.seed,
        'evaluate': not args.no_eval,
        'plot_results': not args.no_plot,
        'save_results': not args.no_save,
        'save_agent': not args.no_save,
        'verbose': not args.quiet,
        'eval_episodes': 100
    }
    
    # Agent-specific configuration
    agent_config = {
        'gamma': args.gamma,
        'epsilon': args.epsilon,
        'eps_decay': args.eps_decay
    }
    
    if args.lr is not None:
        agent_config['lr'] = args.lr
    else:
        agent_config['lr'] = 0.1  # Default learning rate for tabular methods
    
    config['agent_config'] = agent_config
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Run experiment
    try:
        results, agent = run_experiment(config)
        print("\nExperiment completed successfully!")
        
        if not args.quiet:
            print(f"\nFinal performance summary:")
            print(f"Training episodes: {len(results['rewards'])}")
            print(f"Average reward (last 100): {np.mean(results['rewards'][-100:]):.3f}")
            print(f"Success rate: {results['success_rate']:.3f}")
            
            if 'evaluation' in results:
                eval_results = results['evaluation']
                print(f"Evaluation success rate: {eval_results['success_rate']:.3f}")
    
    except Exception as e:
        print(f"Experiment failed with error: {e}")
        raise

if __name__ == "__main__":
    main()