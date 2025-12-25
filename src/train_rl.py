"""
RL Training Script - Train the Supernode RL agent offline.

Based on: Sutton & Barto, "Reinforcement Learning: An Introduction" (2020)

This script trains the PPO agent for sufficient timesteps and saves
the model for later use in tournaments. Training is separated from
evaluation for efficiency.

Usage:
    python train_rl.py --timesteps 50000 --save-path models/ppo_vrp
"""

import argparse
import os
from environment import ProblemInstance
from miner_rl import SupernodeRL, VRPEnvironment, SB3_AVAILABLE

if not SB3_AVAILABLE:
    print("ERROR: stable-baselines3 not installed!")
    print("Install with: pip install stable-baselines3 gymnasium")
    exit(1)

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import torch


def train_rl_agent(timesteps=50000, save_path="models/ppo_vrp", seed=42):
    """
    Train the RL agent with sufficient timesteps for learning.
    
    Args:
        timesteps: Number of training timesteps (recommend 50000+)
        save_path: Path to save the trained model
        seed: Random seed for reproducibility
    """
    print("=" * 60)
    print("RL AGENT TRAINING")
    print("=" * 60)
    
    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    # Create training environment
    print("\nCreating environment...")
    problem = ProblemInstance(num_nodes=20, random_seed=seed, k_neighbors=5)
    env = VRPEnvironment(problem)
    
    print(f"  Nodes: {problem.graph.number_of_nodes()}")
    print(f"  Packages: {len(problem.packages)}")
    print(f"  Edges: {problem.graph.number_of_edges()}")
    
    # Create model directory
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else "models", exist_ok=True)
    
    # Initialize PPO with good hyperparameters for VRP
    print(f"\nInitializing PPO...")
    print(f"  Timesteps: {timesteps:,}")
    
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Encourage exploration
        device=device,
    )
    
    # Train
    print(f"\nTraining for {timesteps:,} timesteps...")
    print("This may take a few minutes...\n")
    
    model.learn(total_timesteps=timesteps)
    
    # Save model
    model.save(save_path)
    print(f"\n[SUCCESS] Model saved to: {save_path}.zip")
    
    # Test the trained model
    print("\n" + "=" * 60)
    print("TESTING TRAINED MODEL")
    print("=" * 60)
    
    obs, _ = env.reset()
    total_reward = 0
    steps = 0
    max_steps = 200
    
    while steps < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        if hasattr(action, 'item'):
            action = action.item()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        
        if terminated:
            break
    
    delivered = len(problem.packages) - len(env.pending_packages)
    print(f"  Packages delivered: {delivered}/{len(problem.packages)}")
    print(f"  Total cost: {env.total_cost:.2f}")
    print(f"  Steps: {steps}")
    print(f"  Total reward: {total_reward:.2f}")
    
    if delivered == len(problem.packages):
        print("\n[SUCCESS] Agent successfully learned to deliver all packages!")
    else:
        print(f"\n[WARNING] Agent only delivered {delivered} packages. Consider training longer.")
    
    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL agent for PoUW")
    parser.add_argument("--timesteps", type=int, default=50000,
                       help="Training timesteps (default: 50000)")
    parser.add_argument("--save-path", type=str, default="models/ppo_vrp",
                       help="Path to save model (default: models/ppo_vrp)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    
    args = parser.parse_args()
    train_rl_agent(args.timesteps, args.save_path, args.seed)
