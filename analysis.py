import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import json
import pandas as pd
from pathlib import Path
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment.gridworld import GridWorld, make_simple_grid, make_maze_grid
from agents.q_table import QTableAgent, SarsaAgent, ExpectedSarsaAgent
from agents.double_q import DoubleQAgent
from train import run_experiment


class TDLearningAnalyzer:
    def __init__(self, results_dir='results', save_plots=False, no_display=False):
        self.results_dir = Path(results_dir)
        self.save_plots = save_plots
        self.no_display = no_display
        self.all_results = {}
        self.performance_summary = {}
        self._setup_plotting()
        
    def _setup_plotting(self):
        plt.style.use('default')
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 11
        if self.no_display:
            plt.ioff()
        else:
            plt.ion()
    
    def load_results(self):
        print(f"Loading results from {self.results_dir}...")
        results_data = {}
        
        if not self.results_dir.exists():
            print(f"Results directory {self.results_dir} does not exist!")
            return results_data
            
        for file_path in self.results_dir.glob('*.json'):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Extract info from filename
                filename = file_path.stem
                parts = filename.split('_')
                
                if len(parts) >= 3:
                    agent_type = parts[0]
                    env_name = parts[1]
                    timestamp = '_'.join(parts[2:])
                    
                    key = f"{agent_type}_{env_name}_{timestamp}"
                    results_data[key] = {
                        'agent_type': agent_type,
                        'env_name': env_name,
                        'timestamp': timestamp,
                        'config': data['config'],
                        'results': data['results']
                    }
                    
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        self.all_results = results_data
        print(f"Loaded {len(results_data)} result files:")
        for key in results_data.keys():
            print(f"  - {key}")
        
        return results_data
    
    def train_sample_agents_if_needed(self):
        if len(self.all_results) > 0:
            return
            
        print("No results found. Training sample agents for demonstration...")
        
        # Common configuration
        base_config = {
            'episodes': 500,
            'seed': 42,
            'save_results': False,
            'save_agent': False,
            'plot_results': False,
            'verbose': False,
            'agent_config': {'lr': 0.1, 'gamma': 0.99, 'epsilon': 1.0, 'eps_decay': 0.995}
        }
        
        # Agent configurations
        agent_configs = [
            {'agent_type': 'qtable', 'env_name': 'standard'},
            {'agent_type': 'sarsa', 'env_name': 'standard'},
            {'agent_type': 'expected_sarsa', 'env_name': 'standard'},
            {'agent_type': 'double_q', 'env_name': 'standard'}
        ]
        
        sample_results = {}
        
        for config in agent_configs:
            full_config = {**base_config, **config}
            print(f"Training {config['agent_type']} agent...")
            
            results, agent = run_experiment(full_config)
            
            key = f"{config['agent_type']}_simple"
            sample_results[key] = {
                'agent_type': config['agent_type'],
                'env_name': 'standard',
                'results': results,
                'config': full_config,
                'agent': agent
            }
        
        self.all_results = sample_results
        print("Sample training completed for all four TD learning algorithms!")
    
    def plot_learning_curve(self, results, title, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        
        rewards = results['rewards']
        episodes = range(len(rewards))
        
        # Plot raw rewards with transparency
        ax.plot(episodes, rewards, alpha=0.3, color='blue', label='Episode Reward')
        
        # Plot moving average
        window_size = max(1, len(rewards) // 20)
        if len(rewards) >= window_size:
            moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            ax.plot(range(len(moving_avg)), moving_avg, color='red', linewidth=2, 
                   label=f'Moving Average ({window_size})')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def analyze_individual_agents(self):
        print("\nIndividual Agent Analysis")
        print("-" * 30)
        
        if not self.all_results:
            print("No results to analyze!")
            return
        
        # Plot learning curves
        n_results = len(self.all_results)
        cols = min(2, n_results)
        rows = (n_results + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if n_results == 1:
            axes = [axes]
        elif rows == 1 and cols > 1:
            axes = axes.reshape(1, -1)
        
        for i, (key, data) in enumerate(self.all_results.items()):
            if rows == 1 and cols == 1:
                ax = axes
            elif rows == 1:
                ax = axes[i]
            else:
                row, col = i // cols, i % cols
                ax = axes[row, col] if isinstance(axes[0], np.ndarray) else axes[row]
            
            title = f"{data['agent_type'].upper()} - {data['env_name']}"
            self.plot_learning_curve(data['results'], title, ax)
        
        # Hide empty subplots
        if isinstance(axes, np.ndarray):
            flat_axes = axes.flatten() if axes.ndim > 1 else axes
            for i in range(n_results, len(flat_axes)):
                flat_axes[i].set_visible(False)
        
        plt.tight_layout()
        if self.save_plots:
            plt.savefig(self.results_dir / 'individual_learning_curves.png', dpi=300, bbox_inches='tight')
        if not self.no_display:
            plt.show()
        
        # Analyze performance metrics
        self.performance_summary = {}
        for key, data in self.all_results.items():
            agent_name = f"{data['agent_type']} ({data['env_name']})"
            metrics = self._analyze_agent_performance(data['results'], agent_name)
            self.performance_summary[agent_name] = metrics
            
            print(f"\nPerformance Analysis: {agent_name}")
            print("-" * 50)
            print(f"Training Episodes: {metrics['total_episodes']}")
            print(f"Training Time: {metrics['training_time']:.2f} seconds")
            print(f"Final Success Rate: {metrics['final_success_rate']:.3f}")
            print(f"Mean Reward: {metrics['mean_reward']:.3f} ± {metrics['std_reward']:.3f}")
            print(f"Best Episode Reward: {metrics['best_reward']:.3f}")
            print(f"Mean Episode Length: {metrics['mean_episode_length']:.1f}")
            
            if metrics['first_success'] is not None:
                print(f"First Success: Episode {metrics['first_success']}")
            else:
                print("First Success: Never achieved")
            
            if metrics.get('episode_to_50pct_success'):
                print(f"50% Success Rate: Episode {metrics['episode_to_50pct_success']}")
            
            if 'recent_mean_reward' in metrics:
                print(f"Recent Performance (last 100): {metrics['recent_mean_reward']:.3f} ± {metrics['recent_std_reward']:.3f}")
                print(f"Recent Success Rate: {metrics['recent_success_rate']:.3f}")
            
            print(f"Final Exploration Rate: {metrics['final_epsilon']:.4f}")
    
    def _analyze_agent_performance(self, results, agent_name):
        rewards = np.array(results['rewards'])
        episode_lengths = np.array(results['episode_lengths'])
        
        # Calculate performance metrics
        metrics = {
            'total_episodes': len(rewards),
            'final_success_rate': results['success_rate'],
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'best_reward': np.max(rewards),
            'worst_reward': np.min(rewards),
            'mean_episode_length': np.mean(episode_lengths),
            'std_episode_length': np.std(episode_lengths),
            'training_time': results['training_time'],
            'final_epsilon': results['final_epsilon']
        }
        
        # Learning progress analysis
        success_episodes = np.where(rewards > 0.5)[0]
        if len(success_episodes) > 0:
            metrics['first_success'] = success_episodes[0]
            
            # Find episode when 50% success rate achieved
            window_size = 100
            for i in range(window_size, len(rewards)):
                recent_success_rate = np.mean(rewards[i-window_size:i] > 0.5)
                if recent_success_rate >= 0.5:
                    metrics['episode_to_50pct_success'] = i
                    break
        else:
            metrics['first_success'] = None
            metrics['episode_to_50pct_success'] = None
        
        # Learning stability (last 100 episodes)
        if len(rewards) >= 100:
            recent_rewards = rewards[-100:]
            metrics['recent_mean_reward'] = np.mean(recent_rewards)
            metrics['recent_std_reward'] = np.std(recent_rewards)
            metrics['recent_success_rate'] = np.mean(recent_rewards > 0.5)
        
        return metrics
    
    def compare_agents(self):
        print("\nAgent Comparison")
        print("-" * 20)
        
        if len(self.all_results) < 2:
            print("Need at least 2 agents for comparison.")
            return
        
        # Learning curves comparison
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.all_results)))
        
        # Learning curves
        ax1 = axes[0]
        for i, (key, data) in enumerate(self.all_results.items()):
            rewards = data['results']['rewards']
            
            # Plot smoothed curve
            window_size = max(1, len(rewards) // 30)
            if len(rewards) >= window_size:
                moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
                label = f"{data['agent_type'].upper()} ({data['env_name']})"
                ax1.plot(range(len(moving_avg)), moving_avg, 
                        color=colors[i], linewidth=2, label=label)
        
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Average Reward')
        ax1.set_title('Learning Curves Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Success rates comparison
        ax2 = axes[1]
        window_size = 50
        
        for i, (key, data) in enumerate(self.all_results.items()):
            rewards = np.array(data['results']['rewards'])
            success_rates = []
            
            for j in range(len(rewards)):
                start_idx = max(0, j - window_size + 1)
                recent_rewards = rewards[start_idx:j+1]
                success_rate = np.mean(recent_rewards > 0.5)
                success_rates.append(success_rate)
            
            label = f"{data['agent_type'].upper()} ({data['env_name']})"
            ax2.plot(success_rates, color=colors[i], linewidth=2, label=label)
        
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Success Rate')
        ax2.set_title(f'Success Rates (Rolling Window: {window_size})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        if self.save_plots:
            plt.savefig(self.results_dir / 'agent_comparison.png', dpi=300, bbox_inches='tight')
        if not self.no_display:
            plt.show()
        
        # Performance comparison table
        self._create_performance_table()
        
        # Performance metrics bar charts
        self._plot_performance_metrics()
    
    def _create_performance_table(self):
        if len(self.performance_summary) == 0:
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(self.performance_summary).T
        
        # Select key metrics for comparison
        comparison_metrics = [
            'final_success_rate', 'mean_reward', 'first_success', 
            'episode_to_50pct_success', 'mean_episode_length', 'training_time'
        ]
        
        comparison_df = df[comparison_metrics].round(3)
        comparison_df.columns = [
            'Success Rate', 'Mean Reward', 'First Success (Ep)', 
            '50% Success (Ep)', 'Avg Length', 'Training Time (s)'
        ]
        
        print("\n📈 Performance Comparison Table")
        print("=" * 80)
        print(comparison_df.to_string())
        
        # Highlight best performers
        print("\nBest Performers:")
        print(f"Highest Success Rate: {comparison_df['Success Rate'].idxmax()} ({comparison_df['Success Rate'].max():.3f})")
        print(f"Highest Mean Reward: {comparison_df['Mean Reward'].idxmax()} ({comparison_df['Mean Reward'].max():.3f})")
        print(f"Fastest Training: {comparison_df['Training Time (s)'].idxmin()} ({comparison_df['Training Time (s)'].min():.2f}s)")
        
        first_success_valid = comparison_df['First Success (Ep)'].dropna()
        if len(first_success_valid) > 0:
            fastest_learner = first_success_valid.idxmin()
            print(f"Fastest First Success: {fastest_learner} (Episode {first_success_valid.min():.0f})")
    
    def _plot_performance_metrics(self):
        if len(self.performance_summary) < 2:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Performance Metrics Comparison', fontsize=16)
        
        agents = list(self.performance_summary.keys())
        colors = plt.cm.Set2(np.linspace(0, 1, len(agents)))
        
        # Success Rate
        success_rates = [self.performance_summary[agent]['final_success_rate'] for agent in agents]
        axes[0, 0].bar(agents, success_rates, color=colors)
        axes[0, 0].set_title('Final Success Rate')
        axes[0, 0].set_ylabel('Success Rate')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Mean Reward
        mean_rewards = [self.performance_summary[agent]['mean_reward'] for agent in agents]
        axes[0, 1].bar(agents, mean_rewards, color=colors)
        axes[0, 1].set_title('Mean Episode Reward')
        axes[0, 1].set_ylabel('Mean Reward')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Training Time
        training_times = [self.performance_summary[agent]['training_time'] for agent in agents]
        axes[1, 0].bar(agents, training_times, color=colors)
        axes[1, 0].set_title('Training Time')
        axes[1, 0].set_ylabel('Time (seconds)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Episode Length
        episode_lengths = [self.performance_summary[agent]['mean_episode_length'] for agent in agents]
        axes[1, 1].bar(agents, episode_lengths, color=colors)
        axes[1, 1].set_title('Mean Episode Length')
        axes[1, 1].set_ylabel('Steps')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        if self.save_plots:
            plt.savefig(self.results_dir / 'performance_metrics.png', dpi=300, bbox_inches='tight')
        if not self.no_display:
            plt.show()
    
    def statistical_analysis(self):
        print("\nBasic Statistical Comparison")
        print("-" * 25)
        
        if len(self.all_results) < 2:
            print("Need at least 2 agents for comparison.")
            return
        
        # Collect reward data
        reward_data = {}
        for key, data in self.all_results.items():
            agent_name = f"{data['agent_type']} ({data['env_name']})"
            reward_data[agent_name] = np.array(data['results']['rewards'])
        
        agent_names = list(reward_data.keys())
        
        # Pairwise comparisons
        print("\nPairwise Statistical Tests (Mann-Whitney U Test):")
        print("-" * 60)
        
        for i in range(len(agent_names)):
            for j in range(i+1, len(agent_names)):
                agent1, agent2 = agent_names[i], agent_names[j]
                rewards1, rewards2 = reward_data[agent1], reward_data[agent2]
                
                # Mann-Whitney U test (non-parametric)
                statistic, p_value = stats.mannwhitneyu(rewards1, rewards2, alternative='two-sided')
                
                # Effect size (Cliff's Delta approximation)
                effect_size = (np.median(rewards1) - np.median(rewards2)) / np.std(np.concatenate([rewards1, rewards2]))
                
                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                
                print(f"{agent1} vs {agent2}:")
                print(f"  p-value: {p_value:.4f} {significance}")
                print(f"  Effect size: {effect_size:.3f}")
                print(f"  Medians: {np.median(rewards1):.3f} vs {np.median(rewards2):.3f}")
                print()
        
        # Learning curve convergence analysis
        print("Learning Curve Analysis:")
        print("-" * 30)
        
        for agent_name, rewards in reward_data.items():
            # Calculate convergence metrics
            if len(rewards) >= 100:
                early_rewards = rewards[:100]
                late_rewards = rewards[-100:]
                
                improvement = np.mean(late_rewards) - np.mean(early_rewards)
                stability = np.std(late_rewards)
                
                print(f"{agent_name}:")
                print(f"  Learning Improvement: {improvement:+.3f}")
                print(f"  Final Stability (std): {stability:.3f}")
                print(f"  Convergence Score: {improvement / (stability + 0.001):.2f}")
                print()
    
    def generate_insights(self):
        print("\nResults Summary")
        print("-" * 20)
        
        if len(self.all_results) == 0:
            print("No results available for analysis.")
            return
        
        agent_types = {}
        for key, data in self.all_results.items():
            agent_type = data['agent_type']
            if agent_type not in agent_types:
                agent_types[agent_type] = []
            agent_types[agent_type].append((key, data['results']))
        
        print(f"Agent Performance Comparison ({len(self.all_results)} experiments):")
        performance_summary = []
        
        for agent_type, experiments in agent_types.items():
            avg_success = np.mean([results['success_rate'] for _, results in experiments])
            avg_reward = np.mean([np.mean(results['rewards'][-100:]) for _, results in experiments])
            avg_time = np.mean([results['training_time'] for _, results in experiments])
            
            performance_summary.append({
                'agent': agent_type,
                'success_rate': avg_success,
                'avg_reward': avg_reward,
                'training_time': avg_time,
                'count': len(experiments)
            })
            
            print(f"  {agent_type.upper().replace('_', ' ')}:")
            print(f"    Success Rate: {avg_success:.3f}")
            print(f"    Avg Reward (last 100): {avg_reward:.3f}")
            print(f"    Training Time: {avg_time:.2f}s")
            print(f"    Experiments: {len(experiments)}")
        
        # Sort by success rate
        performance_summary.sort(key=lambda x: x['success_rate'], reverse=True)
        
        print(f"\nRanking by Success Rate:")
        for i, perf in enumerate(performance_summary, 1):
            print(f"  {i}. {perf['agent'].replace('_', ' ').title()}: {perf['success_rate']:.3f}")
        
        # Environment complexity insights
        env_performance = {}
        for key, data in self.all_results.items():
            env_name = data['env_name']
            if env_name not in env_performance:
                env_performance[env_name] = []
            env_performance[env_name].append(data['results']['success_rate'])
        
        if len(env_performance) > 1:
            print("\n🌍 Environment Difficulty Analysis:")
            for env_name, success_rates in env_performance.items():
                avg_success = np.mean(success_rates)
                print(f"  {env_name.capitalize()}: {avg_success:.3f} average success rate")
        
        # General recommendations based on TD learning theory
        print("\n💡 TD Learning Insights & Recommendations:")
        print("  • Q-Learning: Off-policy, learns optimal policy regardless of behavior")
        print("  • SARSA: On-policy, safer in stochastic environments")
        print("  • Expected SARSA: Combines benefits of Q-Learning and SARSA")
        print("  • Double Q-Learning: Reduces maximization bias in Q-value estimation")
        print("  • All methods converge to optimal policy under tabular conditions")
        print("  • Performance differences often due to exploration vs exploitation balance")
        print("  • Consider environment stochasticity when choosing algorithm")
    
    def create_summary_report(self):
        print("\nSummary Report")
        print("FINAL SUMMARY REPORT")
        print("="*60)
        
        report = f"""
RL GridWorld: Temporal Difference Learning Analysis
================================================

This analysis compared four fundamental temporal difference learning
algorithms on GridWorld environments, focusing on their theoretical
differences and practical performance characteristics.

Experiments conducted: {len(self.all_results)}
Algorithms compared: Q-Learning, SARSA, Expected SARSA, Double Q-Learning
Environments tested: Standard Grid, Maze Grid

Key Findings:
"""
        
        if len(self.all_results) > 0:
            # Add specific findings based on results
            best_agent = max(self.all_results.items(), 
                            key=lambda x: x[1]['results']['success_rate'])
            
            report += f"""
• Best performing agent: {best_agent[1]['agent_type'].upper().replace('_', ' ')}
• Success rate achieved: {best_agent[1]['results']['success_rate']:.3f}
• Total experiments analyzed: {len(self.all_results)}
"""
        
        report += """
Methodological Notes:
• Used epsilon-greedy exploration for all tabular agents
• Implemented standard RL algorithms with consistent parameters
• Consistent hyperparameters across similar agent types
• Environments designed to test different aspects of TD learning

Algorithm Implementations:
• Q-Learning: Off-policy TD control
• SARSA: On-policy TD control
• Expected SARSA: Expected value variant
• Double Q-Learning: Maximization bias reduction

Future Work:
• Test on larger state spaces with function approximation
• Add stochastic environment dynamics
• Conduct more extensive hyperparameter optimization
• Analyze convergence properties in different environments
"""
        
        print(report)
        
        # Save report to file
        if self.save_plots:
            with open(self.results_dir / 'analysis_report.txt', 'w') as f:
                f.write(report)
            print(f"\nReport saved to: {self.results_dir / 'analysis_report.txt'}")
        
        return report
    
    def run_full_analysis(self):
        print("Starting analysis...")
        print("-" * 30)
        
        # Load results
        self.load_results()
        
        # Train sample agents if no results available
        self.train_sample_agents_if_needed()
        
        # Run analyses
        self.analyze_individual_agents()
        self.compare_agents()
        self.statistical_analysis()
        self.generate_insights()
        self.create_summary_report()
        
        print("\n✅ Analysis complete!")
        if self.save_plots:
            print(f"Plots saved to: {self.results_dir}")


def main():
    """Main entry point for the analysis script."""
    parser = argparse.ArgumentParser(description='Analyze TD Learning experiments')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Directory containing result files')
    parser.add_argument('--save-plots', action='store_true',
                       help='Save plots to files')
    parser.add_argument('--no-display', action='store_true',
                       help='Do not display plots (useful for headless environments)')
    
    args = parser.parse_args()
    
    # Create analyzer and run analysis
    analyzer = TDLearningAnalyzer(
        results_dir=args.results_dir,
        save_plots=args.save_plots,
        no_display=args.no_display
    )
    
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()