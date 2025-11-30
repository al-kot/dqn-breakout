import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import ale_py
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation, TransformReward
import os
import csv
import itertools
from tqdm import tqdm

# Register ALE environments
gym.register_envs(ale_py)

# --- Classes & Functions copied/adapted from notebook ---

def make_env(render_mode=None):
    env = gym.make("ALE/Breakout-v5", render_mode=render_mode, full_action_space=False, repeat_action_probability=0, frameskip=1)
    env = AtariPreprocessing(
        env, 
        screen_size=84,
        grayscale_obs=True,
        frame_skip=4,
        scale_obs=False
    )
    env = TransformReward(env, lambda r: float(np.sign(r)))
    env = FrameStackObservation(env, 4)
    return env

class DQN(nn.Module):
    def __init__(self, n_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU()
        )
        self.fc_input_dim = 32 * 9 * 9
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

    def forward(self, x):
        x = x.float() / 255.0
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        state = np.array(state, dtype=np.uint8)
        next_state = np.array(next_state, dtype=np.uint8)
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            np.array(state),
            np.array(action),
            np.array(reward, dtype=np.float32),
            np.array(next_state),
            np.array(done, dtype=np.uint8)
        )
    
    def __len__(self):
        return len(self.buffer)

# --- Main Training Function ---

def run_training(lr, batch_size, gamma, total_frames=100000, output_file="results.csv"):
    """
    Runs one training session with specific hyperparameters.
    """
    print(f"\n>>> Starting Run: LR={lr}, Batch={batch_size}, Gamma={gamma}")
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env()
    n_actions = env.action_space.n
    
    policy_net = DQN(n_actions).to(device)
    optimizer = optim.RMSprop(policy_net.parameters(), lr=lr, alpha=0.95, eps=0.01)
    
    # Reduced memory size for faster testing/less RAM usage during search
    memory = ReplayBuffer(50000) 
    
    # Hyperparameters fixed for this search logic (but could be parameterized)
    EPS_START = 1.0
    EPS_END = 0.1
    EPS_DECAY_STEPS = total_frames * 0.8 # Decay over 80% of training
    START_LEARNING = 1000 # Start learning quickly for this test
    
    steps_done = 0
    episode_rewards = []
    
    # Logging setup
    # We will write to CSV incrementally
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Step", "Episode", "Reward", "Epsilon", "LR", "BatchSize", "Gamma"])

    state, _ = env.reset()
    current_ep_reward = 0
    episode_count = 0

    # Progress bar
    bar = tqdm(range(total_frames), desc=f"LR={lr} BS={batch_size}")
    
    for step in bar:
        # Epsilon
        epsilon = np.interp(step, [0, EPS_DECAY_STEPS], [EPS_START, EPS_END])
        
        # Select Action
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                state_t = torch.tensor(state, device=device).unsqueeze(0)
                q_values = policy_net(state_t)
                action = q_values.argmax().item()
        
        # Step
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        memory.push(state, action, reward, next_state, done)
        state = next_state
        current_ep_reward += reward
        
        # Optimize
        if step > START_LEARNING and len(memory) >= batch_size:
            # Sample
            s_batch, a_batch, r_batch, ns_batch, d_batch = memory.sample(batch_size)
            
            s_t = torch.tensor(s_batch, device=device)
            a_t = torch.tensor(a_batch, device=device).unsqueeze(1)
            r_t = torch.tensor(r_batch, device=device)
            ns_t = torch.tensor(ns_batch, device=device)
            d_t = torch.tensor(d_batch, device=device)
            
            # Q(s,a)
            q_vals = policy_net(s_t).gather(1, a_t)
            
            # V(s')
            with torch.no_grad():
                next_q_vals = policy_net(ns_t).max(1)[0]
            
            # Target
            expected_q_vals = r_t + (gamma * next_q_vals * (1 - d_t))
            
            loss = nn.functional.mse_loss(q_vals.squeeze(), expected_q_vals)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # End of Episode
        if done:
            state, _ = env.reset()
            episode_count += 1
            episode_rewards.append(current_ep_reward)
            
            # Log to CSV
            with open(output_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([step, episode_count, current_ep_reward, epsilon, lr, batch_size, gamma])
            
            bar.set_description(f"LR={lr} BS={batch_size} | EpRew={current_ep_reward:.1f}")
            current_ep_reward = 0
            
    env.close()
    print(f"Finished run. Results saved to {output_file}")

# --- Hyperparameter Grid Search ---

if __name__ == "__main__":
    # Define parameters to test
    # We test 3 important parameters:
    # 1. Learning Rate (Critical for convergence speed and stability)
    # 2. Batch Size (Affects stability of gradient updates)
    # 3. Gamma (Discount factor, affects long-term planning)
    
    learning_rates = [0.0001, 0.00025, 0.0005]
    batch_sizes = [32, 64]
    # gammas = [0.99] # Keeping gamma constant to reduce search space, but can be added
    
    # Total frames per run (Reduced for demonstration purposes)
    # In a real experiment, this should be 2M - 10M
    FRAMES_PER_RUN = 200000 
    
    os.makedirs("grid_search_results", exist_ok=True)
    
    combinations = list(itertools.product(learning_rates, batch_sizes))
    print(f"Starting Grid Search with {len(combinations)} combinations...")
    
    for lr, bs in combinations:
        gamma = 0.99 # Default
        
        # Define unique filename for this run
        filename = f"grid_search_results/dqn_lr{lr}_bs{bs}_g{gamma}.csv"
        
        run_training(lr, bs, gamma, total_frames=FRAMES_PER_RUN, output_file=filename)

    print("\nAll experiments completed.")
