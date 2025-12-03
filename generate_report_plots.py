import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def generate_plots():
    try:
        df = pd.read_csv('dqn_training_log.csv')
    except FileNotFoundError:
        print("Error: dqn_training_log.csv not found.")
        return

    # Calculate rolling average for rewards to smooth the plot
    window_size = 50
    df['reward_rolling_avg'] = df['episode_reward'].rolling(window=window_size).mean()

    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

    # 1. Episode Reward
    axes[0].plot(df['step'], df['episode_reward'], label='Episode Reward', alpha=0.3, color='lightblue')
    axes[0].plot(df['step'], df['reward_rolling_avg'], label=f'Rolling Avg ({window_size})', color='blue')
    axes[0].set_ylabel('Reward')
    axes[0].set_title('Training Reward over Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Average Loss
    # Loss can be noisy, maybe log scale?
    axes[1].plot(df['step'], df['avg_loss'], label='Avg Loss', color='orange')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Training Loss over Time')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    # axes[1].set_yscale('log')

    # 3. Evaluation Reward
    # Filter rows where eval_reward is not 0 or use a specific marker if it's sparse
    # The csv seems to log eval_reward at every step but it's 0 most of the time?
    # Let's check how eval_reward is logged. In the ipynb:
    # csv_writer.writerow([step, episode_reward, epsilon, avg_loss, best, latest_eval_reward])
    # latest_eval_reward is updated periodically. So it holds the last value.
    # We should plot it as a line.
    axes[2].plot(df['step'], df['eval_reward'], label='Evaluation Reward', color='green')
    axes[2].set_ylabel('Eval Reward')
    axes[2].set_xlabel('Training Steps (Frames)')
    axes[2].set_title('Evaluation Reward over Time')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_metrics.png')
    print("Plots saved to training_metrics.png")

if __name__ == "__main__":
    generate_plots()
