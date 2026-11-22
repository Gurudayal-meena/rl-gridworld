import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
import json
import os
from pathlib import Path

from environment.gridworld import GridWorld, make_simple_grid, make_maze_grid
from agents.q_table import QTableAgent
from agents.double_q import DoubleQAgent

def load_agent(agent_path, env):
    if 'double_q' in agent_path:
        agent = DoubleQAgent(env.n_states, env.n_actions)
    else:
        agent = QTableAgent(env.n_states, env.n_actions)
        agent.load(agent_path)
    
    return agent

def visualize_policy(agent, env, save_path=None):
    policy = agent.get_policy()
    values = agent.get_value_function()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Policy visualization
    ax1.set_title('Learned Policy', fontsize=14)
    ax1.set_xlim(-0.5, env.n_cols - 0.5)
    ax1.set_ylim(-0.5, env.n_rows - 0.5)
    ax1.set_aspect('equal')
    
    # Draw grid
    for i in range(env.n_rows + 1):
        ax1.axhline(y=i-0.5, color='black', linewidth=0.5)
    for j in range(env.n_cols + 1):
        ax1.axvline(x=j-0.5, color='black', linewidth=0.5)
    
    # Action arrows
    action_symbols = ['↑', '→', '↓', '←']
    action_colors = ['red', 'blue', 'green', 'orange']
    
    for state in range(env.n_states):
        r, c = state // env.n_cols, state % env.n_cols
        
        # Color cell based on type
        if (r, c) == env.goal:
            rect = patches.Rectangle((c-0.5, r-0.5), 1, 1, 
                                   facecolor='lightgreen', alpha=0.7)
            ax1.add_patch(rect)
            ax1.text(c, r, 'G', ha='center', va='center', fontsize=16, fontweight='bold')
        elif (r, c) in env.traps:
            rect = patches.Rectangle((c-0.5, r-0.5), 1, 1, 
                                   facecolor='lightcoral', alpha=0.7)
            ax1.add_patch(rect)
            ax1.text(c, r, 'T', ha='center', va='center', fontsize=16, fontweight='bold')
        else:
            action = policy[state]
            ax1.text(c, r, action_symbols[action], ha='center', va='center', 
                    fontsize=20, color=action_colors[action], fontweight='bold')
    
    ax1.set_xticks(range(env.n_cols))
    ax1.set_yticks(range(env.n_rows))
    ax1.invert_yaxis()
    
    # Value function visualization
    ax2.set_title('State Values', fontsize=14)
    value_grid = values.reshape(env.n_rows, env.n_cols)
    
    im = ax2.imshow(value_grid, cmap='viridis', interpolation='nearest')
    
    # Add value text
    for r in range(env.n_rows):
        for c in range(env.n_cols):
            value = value_grid[r, c]
            color = 'white' if value < np.mean(value_grid) else 'black'
            ax2.text(c, r, f'{value:.2f}', ha='center', va='center', 
                    color=color, fontweight='bold')
    
    # Mark special states
    goal_r, goal_c = env.goal
    ax2.scatter(goal_c, goal_r, marker='*', s=200, c='red', edgecolor='white', linewidth=2)
    
    for trap_r, trap_c in env.traps:
        ax2.scatter(trap_c, trap_r, marker='X', s=200, c='red', edgecolor='white', linewidth=2)
    
    ax2.set_xticks(range(env.n_cols))
    ax2.set_yticks(range(env.n_rows))
    plt.colorbar(im, ax=ax2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Policy visualization saved to {save_path}")
    
    return fig

def create_episode_animation(env, agent, save_path=None, max_steps=50):
    # Record episode
    episode, _ = env.record_episode(agent, max_steps)
    
    if len(episode) <= 1:
        print("Episode too short for animation")
        return None
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    def animate_frame(frame_idx):
        ax.clear()
        
        # Draw grid
        for i in range(env.n_rows + 1):
            ax.axhline(y=i, color='black', linewidth=1)
        for j in range(env.n_cols + 1):
            ax.axvline(x=j, color='black', linewidth=1)
        
        # Color special cells
        goal_r, goal_c = env.goal
        goal_rect = patches.Rectangle((goal_c, env.n_rows - goal_r - 1), 1, 1,
                                    facecolor='lightgreen', alpha=0.7)
        ax.add_patch(goal_rect)
        ax.text(goal_c + 0.5, env.n_rows - goal_r - 0.5, 'G', 
               ha='center', va='center', fontsize=20, fontweight='bold')
        
        for trap_r, trap_c in env.traps:
            trap_rect = patches.Rectangle((trap_c, env.n_rows - trap_r - 1), 1, 1,
                                        facecolor='lightcoral', alpha=0.7)
            ax.add_patch(trap_rect)
            ax.text(trap_c + 0.5, env.n_rows - trap_r - 0.5, 'T', 
                   ha='center', va='center', fontsize=20, fontweight='bold')
        
        # Draw agent position
        if frame_idx < len(episode):
            agent_r, agent_c = episode[frame_idx]['position']
            agent_circle = patches.Circle((agent_c + 0.5, env.n_rows - agent_r - 0.5), 
                                        0.3, facecolor='yellow', edgecolor='black', linewidth=2)
            ax.add_patch(agent_circle)
            
            # Draw trajectory up to current step
            for i in range(min(frame_idx + 1, len(episode) - 1)):
                r1, c1 = episode[i]['position']
                r2, c2 = episode[i + 1]['position']
                ax.plot([c1 + 0.5, c2 + 0.5], 
                       [env.n_rows - r1 - 0.5, env.n_rows - r2 - 0.5],
                       'b-', alpha=0.6, linewidth=2)
        
        ax.set_xlim(0, env.n_cols)
        ax.set_ylim(0, env.n_rows)
        ax.set_aspect('equal')
        ax.set_title(f'Episode Step: {frame_idx}/{len(episode)-1}', fontsize=16)
        ax.set_xticks(range(env.n_cols + 1))
        ax.set_yticks(range(env.n_rows + 1))
    
    # Create animation
    anim = FuncAnimation(fig, animate_frame, frames=len(episode), 
                        interval=500, repeat=True, blit=False)
    
    if save_path:
        writer = PillowWriter(fps=2)
        anim.save(save_path, writer=writer)
        print(f"Episode animation saved to {save_path}")
    
    return anim

def compare_agents(results_list, agent_names, save_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Agent Comparison', fontsize=16)
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(results_list)))
    
    # Learning curves
    ax = axes[0, 0]
    for i, (results, name) in enumerate(zip(results_list, agent_names)):
        rewards = results['rewards']
        ax.plot(rewards, alpha=0.3, color=colors[i])
        
        # Smooth curve
        window = max(1, len(rewards) // 50)
        if len(rewards) >= window:
            smooth_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax.plot(range(len(smooth_rewards)), smooth_rewards, 
                   color=colors[i], linewidth=2, label=name)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('Learning Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Success rates
    ax = axes[0, 1]
    window_size = 100
    for i, (results, name) in enumerate(zip(results_list, agent_names)):
        rewards = results['rewards']
        success_rates = []
        for j in range(len(rewards)):
            start_idx = max(0, j - window_size)
            recent_rewards = rewards[start_idx:j+1]
            success_rate = sum(1 for r in recent_rewards if r > 0.5) / len(recent_rewards)
            success_rates.append(success_rate)
        
        ax.plot(success_rates, color=colors[i], linewidth=2, label=name)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Success Rate')
    ax.set_title(f'Success Rates (Rolling Window: {window_size})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # Final performance comparison
    ax = axes[1, 0]
    final_success_rates = [results['success_rate'] for results in results_list]
    bars = ax.bar(agent_names, final_success_rates, color=colors)
    ax.set_ylabel('Final Success Rate')
    ax.set_title('Final Performance Comparison')
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, rate in zip(bars, final_success_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{rate:.3f}', ha='center', va='bottom')
    
    # Training time comparison
    ax = axes[1, 1]
    training_times = [results['training_time'] for results in results_list]
    bars = ax.bar(agent_names, training_times, color=colors)
    ax.set_ylabel('Training Time (seconds)')
    ax.set_title('Training Time Comparison')
    
    # Add value labels on bars
    for bar, time in zip(bars, training_times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{time:.1f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    
    return fig

def analyze_q_table(agent, env, save_path=None):
    if not hasattr(agent, 'q_table'):
        print("Agent doesn't have a Q-table")
        return None
    
    q_table = agent.q_table
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Q-Table Analysis', fontsize=16)
    
    # Q-values heatmap for each action
    action_names = ['Up', 'Right', 'Down', 'Left']
    
    for action in range(4):
        ax = axes[action // 2, action % 2]
        
        q_values = q_table[:, action].reshape(env.n_rows, env.n_cols)
        im = ax.imshow(q_values, cmap='RdYlBu_r', interpolation='nearest')
        
        # Add text annotations
        for r in range(env.n_rows):
            for c in range(env.n_cols):
                value = q_values[r, c]
                color = 'white' if abs(value) < np.std(q_values) else 'black'
                ax.text(c, r, f'{value:.2f}', ha='center', va='center', 
                       color=color, fontsize=8)
        
        ax.set_title(f'{action_names[action]} Action Q-Values')
        ax.set_xticks(range(env.n_cols))
        ax.set_yticks(range(env.n_rows))
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Q-table analysis saved to {save_path}")
    
    return fig

def analyze_double_q_bias(agent, env, save_path=None):
    """
    Analyze Double Q-Learning bias reduction properties.
    """
    if not hasattr(agent, 'get_q_tables'):
        print("Agent doesn't support Double Q analysis")
        return None
    
    q1, q2 = agent.get_q_tables()
    bias_analysis = agent.get_bias_analysis()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Double Q-Learning Bias Analysis', fontsize=16)
    
    # Q1 vs Q2 comparison
    ax = axes[0, 0]
    ax.scatter(q1.flatten(), q2.flatten(), alpha=0.6)
    ax.plot([q1.min(), q1.max()], [q1.min(), q1.max()], 'r--', label='Perfect Agreement')
    ax.set_xlabel('Q1 Values')
    ax.set_ylabel('Q2 Values')
    ax.set_title('Q1 vs Q2 Value Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Bias over time
    if len(agent.training_stats['q_value_stats']) > 0:
        ax = axes[0, 1]
        bias_history = [stat['bias_estimate'] for stat in agent.training_stats['q_value_stats']]
        ax.plot(bias_history, linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Bias Estimate (|Q1 - Q2|)')
        ax.set_title('Bias Reduction Over Time')
        ax.grid(True, alpha=0.3)
    
    # Q-table differences heatmap
    ax = axes[1, 0]
    q_diff = np.abs(q1 - q2).mean(axis=1).reshape(env.n_rows, env.n_cols)
    im = ax.imshow(q_diff, cmap='Reds', interpolation='nearest')
    ax.set_title('Average |Q1 - Q2| by State')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    plt.colorbar(im, ax=ax)
    
    # Update statistics
    ax = axes[1, 1]
    if bias_analysis:
        stats_labels = ['Q1 Updates', 'Q2 Updates']
        stats_values = [bias_analysis['q1_updates_ratio'], bias_analysis['q2_updates_ratio']]
        bars = ax.bar(stats_labels, stats_values, color=['blue', 'red'], alpha=0.7)
        ax.set_ylabel('Update Ratio')
        ax.set_title('Q-Table Update Distribution')
        ax.set_ylim(0, 1)
        
        # Add value labels
        for bar, value in zip(bars, stats_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Double Q bias analysis saved to {save_path}")
    
    return fig

def generate_performance_report(results, agent_name, env_name, save_path=None):
    report = f"""
    Performance Report: {agent_name} on {env_name} Environment
    ================================================
    
    Training Configuration:
    - Episodes: {len(results['rewards'])}
    - Training Time: {results['training_time']:.2f} seconds
    - Final Exploration Rate: {results['final_epsilon']:.4f}
    
    Performance Metrics:
    - Final Success Rate: {results['success_rate']:.3f}
    - Average Reward (last 100 episodes): {np.mean(results['rewards'][-100:]):.3f}
    - Best Episode Reward: {np.max(results['rewards']):.3f}
    - Worst Episode Reward: {np.min(results['rewards']):.3f}
    - Average Episode Length: {np.mean(results['episode_lengths']):.1f}
    
    Learning Progress:
    - Episodes to first success: {next((i for i, r in enumerate(results['rewards']) if r > 0.5), 'None')}
    - Episodes to 50% success rate: {next((i for i in range(100, len(results['rewards'])) if sum(1 for r in results['rewards'][i-100:i] if r > 0.5) >= 50), 'None')}
    - Success rate improvement: {results['success_rate'] - (1 if results['rewards'][0] > 0.5 else 0):.3f}
    """
    
    if 'evaluation' in results:
        eval_results = results['evaluation']
        report += f"""
    Evaluation Results (100 episodes, no exploration):
    - Mean Reward: {eval_results['mean_reward']:.3f} ± {eval_results['std_reward']:.3f}
    - Success Rate: {eval_results['success_rate']:.3f}
    - Mean Episode Length: {eval_results['mean_length']:.1f}
        """
    
    if 'losses' in results and results['losses']:
        report += f"""
    Training Dynamics:
    - Total Training Steps: {len(results['losses'])}
    - Average Loss (last 100 steps): {np.mean(results['losses'][-100:]):.6f}
    - Loss Reduction: {np.mean(results['losses'][:100]) - np.mean(results['losses'][-100:]):.6f}
        """
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
        print(f"Performance report saved to {save_path}")
    
    return report

def main():
    parser = argparse.ArgumentParser(description='Evaluate and visualize RL agents')
    parser.add_argument('--agent-path', type=str, required=True, 
                       help='Path to trained agent file')
    parser.add_argument('--env', type=str, choices=['simple', 'maze'], 
                       default='simple', help='Environment type')
    parser.add_argument('--output-dir', type=str, default='results/evaluation', 
                       help='Output directory for visualizations')
    parser.add_argument('--policy', action='store_true', 
                       help='Visualize policy and value function')
    parser.add_argument('--animate', action='store_true', 
                       help='Create episode animation')
    parser.add_argument('--qtable', action='store_true', 
                       help='Analyze Q-table (for tabular agents)')
    parser.add_argument('--double-q-bias', action='store_true',
                       help='Analyze Double Q-Learning bias reduction')
    parser.add_argument('--report', action='store_true', 
                       help='Generate performance report')
    parser.add_argument('--episodes', type=int, default=5, 
                       help='Number of episodes to animate')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create environment
    env_creators = {
        'simple': make_simple_grid,
        'maze': make_maze_grid
    }
    env = env_creators[args.env]()
    
    # Load agent
    print(f"Loading agent from {args.agent_path}")
    agent = load_agent(args.agent_path, env)
    
    # Get agent name from path
    agent_name = Path(args.agent_path).stem
    
    # Load results if available
    results_path = args.agent_path.replace('_agent.pth', '.json').replace('_agent.pkl', '.json')
    results = None
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            data = json.load(f)
            results = data['results']
        print(f"Loaded results from {results_path}")
    
    # Visualize policy
    if args.policy:
        policy_path = os.path.join(args.output_dir, f"{agent_name}_policy.png")
        visualize_policy(agent, env, policy_path)
    
    # Create animations
    if args.animate:
        for i in range(args.episodes):
            anim_path = os.path.join(args.output_dir, f"{agent_name}_episode_{i+1}.gif")
            create_episode_animation(env, agent, anim_path)
    
    # Analyze Q-table
    if args.qtable and hasattr(agent, 'q_table'):
        qtable_path = os.path.join(args.output_dir, f"{agent_name}_qtable.png")
        analyze_q_table(agent, env, qtable_path)
    
    # Analyze Double Q bias
    if args.double_q_bias and hasattr(agent, 'get_q_tables'):
        bias_path = os.path.join(args.output_dir, f"{agent_name}_double_q_bias.png")
        analyze_double_q_bias(agent, env, bias_path)
    
    # Generate report
    if args.report and results:
        report_path = os.path.join(args.output_dir, f"{agent_name}_report.txt")
        generate_performance_report(results, agent_name, args.env, report_path)
    
    print(f"Evaluation complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()