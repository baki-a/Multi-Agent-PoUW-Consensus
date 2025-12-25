"""
RL Miner (Supernode) - Handles ALL 3 Layers using PPO

Uses Stable-Baselines3 PPO to learn optimal routing while handling:
- Layer 1: Base graph navigation
- Layer 2: Stochastic traffic uncertainty  
- Layer 3: Adversarial competition

This is the most sophisticated miner, capable of generalization across maps.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from miners import MinerAbstract

# Try importing stable-baselines3
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("Warning: stable-baselines3 not installed. Install with: pip install stable-baselines3")


class VRPEnvironment(gym.Env):
    """
    Gymnasium environment for Vehicle Routing Problem with all 3 layers.
    """
    metadata = {'render_modes': ['human']}
    
    def __init__(self, problem_instance, max_steps=100):
        super().__init__()
        
        self.problem = problem_instance
        self.max_steps = max_steps
        self.num_nodes = problem_instance.graph.number_of_nodes()
        self.nodes = list(problem_instance.graph.nodes())
        self.node_to_idx = {node: idx for idx, node in enumerate(self.nodes)}
        self.idx_to_node = {idx: node for node, idx in self.node_to_idx.items()}
        
        # Action space: move to any node (will use masking for valid moves)
        self.action_space = spaces.Discrete(self.num_nodes)
        
        # Observation space:
        # - Current node (one-hot): num_nodes
        # - Pending packages (binary mask): num_nodes
        # - Adversary position (one-hot): num_nodes
        # - Traffic state (normalized probabilities): num_nodes
        obs_size = self.num_nodes * 4
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(obs_size,), dtype=np.float32
        )
        
        self._reset_state()
    
    def _reset_state(self):
        """Initialize episode state."""
        self.current_node = self.problem.get_start_node()
        self.pending_packages = set(self.problem.packages)
        self.steps = 0
        self.total_cost = 0
        self.env_clone = self.problem.clone()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_obs(), {}
    
    def _get_obs(self):
        """Create observation vector."""
        obs = np.zeros(self.num_nodes * 4, dtype=np.float32)
        
        # Current node (one-hot)
        current_idx = self.node_to_idx[self.current_node]
        obs[current_idx] = 1.0
        
        # Pending packages (binary mask)
        offset = self.num_nodes
        for pkg in self.pending_packages:
            pkg_idx = self.node_to_idx[pkg]
            obs[offset + pkg_idx] = 1.0
        
        # Adversary position (one-hot)
        offset = self.num_nodes * 2
        adv_pos = self.env_clone.get_adversary_position()
        if adv_pos:
            adv_idx = self.node_to_idx[adv_pos]
            obs[offset + adv_idx] = 1.0
        
        # Traffic probabilities (normalized)
        offset = self.num_nodes * 3
        for u, v, data in self.env_clone.graph.edges(data=True):
            traffic_prob = data.get('traffic_prob', 0.0)
            if traffic_prob > 0:
                u_idx = self.node_to_idx[u]
                v_idx = self.node_to_idx[v]
                obs[offset + u_idx] = max(obs[offset + u_idx], traffic_prob)
                obs[offset + v_idx] = max(obs[offset + v_idx], traffic_prob)
        
        return obs
    
    def _get_valid_actions(self):
        """Get mask of valid actions (neighboring nodes)."""
        valid = np.zeros(self.num_nodes, dtype=bool)
        for neighbor in self.env_clone.graph.neighbors(self.current_node):
            valid[self.node_to_idx[neighbor]] = True
        return valid
    
    def step(self, action):
        """
        Execute action and return new state.
        
        Reward shaping based on Sutton & Barto (2020) principles:
        - Potential-based shaping to guide toward packages
        - Sparse rewards for delivery milestones
        """
        self.steps += 1
        
        target_node = self.idx_to_node[action]
        
        # Check if action is valid (neighbor node)
        if not self.env_clone.graph.has_edge(self.current_node, target_node):
            # Invalid move - penalty and stay in place
            return self._get_obs(), -10, False, False, {'invalid': True}
        
        # Check if blocked by adversary
        blocked = self.env_clone.is_blocked_by_adversary(target_node)
        
        if blocked:
            reward = -50  # Penalty for being blocked
            return self._get_obs(), reward, False, False, {'blocked': True}
        
        # Calculate potential BEFORE move (for shaping)
        old_potential = self._get_potential()
        
        # Get stochastic cost (sampled)
        cost = self.env_clone.get_stochastic_cost(self.current_node, target_node, sample=True)
        self.total_cost += cost
        
        # Move to target
        self.current_node = target_node
        
        # Base reward: small step penalty
        reward = -1
        
        # Check for package delivery - BIG reward
        if target_node in self.pending_packages:
            self.pending_packages.remove(target_node)
            reward += 200  # Large bonus for delivery
        
        # Potential-based reward shaping (Sutton & Barto)
        # R' = R + gamma * Phi(s') - Phi(s)
        new_potential = self._get_potential()
        shaping = 0.99 * new_potential - old_potential
        reward += shaping
        
        # Move adversary toward miner
        if self.pending_packages:
            self.env_clone.move_adversary_greedy(
                self.current_node, 
                list(self.pending_packages)[0]
            )
        
        # Check termination
        terminated = len(self.pending_packages) == 0
        truncated = self.steps >= self.max_steps
        
        if terminated:
            reward += 500  # Bonus for completing all deliveries
        
        info = {
            'cost': self.total_cost,
            'remaining': len(self.pending_packages)
        }
        
        return self._get_obs(), reward, terminated, truncated, info
    
    def _get_potential(self):
        """
        Potential function for reward shaping.
        Higher potential when closer to remaining packages.
        """
        if not self.pending_packages:
            return 0
        
        coords = self.problem.get_node_coordinates()
        current_pos = coords[self.current_node]
        
        # Negative distance to nearest package (closer = higher potential)
        min_dist = float('inf')
        for pkg in self.pending_packages:
            pkg_pos = coords[pkg]
            from scipy.spatial import distance as dist_func
            d = dist_func.euclidean(current_pos, pkg_pos)
            min_dist = min(min_dist, d)
        
        return -min_dist  # Negative so closer = higher potential
    
    def render(self):
        print(f"Step {self.steps} | Node: {self.current_node} | "
              f"Remaining: {len(self.pending_packages)} | Cost: {self.total_cost:.2f}")


class SupernodeRL(MinerAbstract):
    """
    RL-based miner using PPO from Stable-Baselines3.
    Capable of handling all 3 layers through learned policy.
    """
    
    def __init__(self, model_path=None, training_timesteps=10000):
        super().__init__(name="Supernode RL (PPO)")
        self.model = None
        self.model_path = model_path
        self.training_timesteps = training_timesteps
        self.is_trained = False
    
    def train(self, problem_instance, timesteps=None):
        """
        Train the PPO model on given problem instance.
        
        Args:
            problem_instance: Environment to train on
            timesteps: Training timesteps (default: self.training_timesteps)
        """
        if not SB3_AVAILABLE:
            print("Error: stable-baselines3 not available for training")
            return False
        
        timesteps = timesteps or self.training_timesteps
        
        print(f"Training PPO for {timesteps} timesteps...")
        
        # Create environment
        env = VRPEnvironment(problem_instance)
        
        # Detect device (GPU if available)
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        if device == "cuda":
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        
        # Initialize PPO with GPU support
        # Use smaller n_steps for more frequent updates with limited timesteps
        self.model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=512,      # Reduced from 2048 for more training iterations
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            device=device,  # Use GPU if available
        )
        
        # Train
        self.model.learn(total_timesteps=timesteps)
        self.is_trained = True
        
        print("Training complete!")
        return True
    
    def save(self, path):
        """Save trained model."""
        if self.model:
            self.model.save(path)
            print(f"Model saved to {path}")
    
    def load(self, path):
        """Load trained model."""
        if SB3_AVAILABLE:
            self.model = PPO.load(path)
            self.is_trained = True
            print(f"Model loaded from {path}")
    
    def solve(self, problem_instance):
        """
        Solve VRP using trained RL policy.
        """
        if not SB3_AVAILABLE:
            print("Error: stable-baselines3 not available")
            return None
        
        # Train if not already trained
        if not self.is_trained:
            print("Model not trained. Training now...")
            self.train(problem_instance)
        
        # Create environment for inference
        env = VRPEnvironment(problem_instance)
        obs, _ = env.reset()
        
        path = [env.current_node]
        done = False
        max_solve_steps = 200  # Safety limit to prevent infinite loops
        step_count = 0
        
        print("Resolviendo problema con RL (PPO - All Layers)...")
        
        while not done and step_count < max_solve_steps:
            step_count += 1
            # Get action from policy
            action, _ = self.model.predict(obs, deterministic=True)
            
            # Convert numpy array to int if needed
            if hasattr(action, 'item'):
                action = action.item()
            else:
                action = int(action)
            
            # Execute action
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Record path
            if env.current_node != path[-1]:
                path.append(env.current_node)
        
        self.solution_path = path
        self.solution_cost = env.total_cost
        
        if len(env.pending_packages) == 0:
            return path
        else:
            print(f"Warning: {len(env.pending_packages)} packages remaining")
            return path


if __name__ == "__main__":
    from environment import ProblemInstance
    
    if not SB3_AVAILABLE:
        print("Please install stable-baselines3: pip install stable-baselines3")
        exit(1)
    
    print("Testing Supernode RL Miner...")
    env = ProblemInstance(num_nodes=15, random_seed=42, k_neighbors=5)
    
    print(f"Start: {env.get_start_node()}")
    print(f"Packages: {env.packages}")
    print(f"Adversary: {env.get_adversary_position()}")
    
    miner = SupernodeRL(training_timesteps=5000)
    
    # Train
    miner.train(env)
    
    # Solve
    path = miner.solve(env)
    
    if path:
        print(f"Solution found: {path}")
        print(f"Total cost: {miner.solution_cost:.2f}")
    else:
        print("No solution found")
