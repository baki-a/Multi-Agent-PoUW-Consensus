"""
Minimax Miner - Handles Layer 1 (Base) + Layer 3 (Adversarial)

Based on: Russell & Norvig, "Artificial Intelligence: A Modern Approach" (4th Ed.)
Chapter 5: Adversarial Search and Games

Uses Minimax with Alpha-Beta Pruning (Figure 5.7 in AIMA) to compete against
an adversary that tries to block the miner's path.

Game formulation:
- Players: Miner (MAX) vs Adversary (MIN)
- State: (miner_pos, adversary_pos, pending_packages)
- Actions: Move to adjacent node
- Terminal: All packages delivered OR miner blocked
- Utility: +1 per package delivered, -penalty for cost
"""

import heapq
import networkx as nx
from miners import MinerAbstract
from heuristic import Heuristic


class NodeMinimax(MinerAbstract):
    """
    Minimax miner implementing adversarial search from AIMA.
    Uses iterative deepening with alpha-beta pruning.
    """
    
    def __init__(self, max_depth=2):
        """
        Initialize Minimax miner.
        
        Args:
            max_depth: Maximum search depth (keep low for tractability)
        """
        super().__init__(name="Miner Minimax")
        self.max_depth = max_depth
        self.nodes_expanded = 0

    def solve(self, problem_instance):
        """
        Solve VRP using greedy best-first search with minimax lookahead.
        
        Per AIMA: We use minimax to evaluate moves, not to plan the entire path,
        since the state space is too large for full game-tree search.
        """
        start_node = problem_instance.get_start_node()
        pending = list(problem_instance.packages)
        
        path = [start_node]
        current = start_node
        total_cost = 0
        
        print(f"Resolviendo problema con Minimax (Layer 1+3)...")
        print(f"  Max depth: {self.max_depth}")
        
        # Clone environment for simulation
        env = problem_instance.clone()
        
        max_steps = len(pending) * 10  # Safety limit
        steps = 0
        
        while pending and steps < max_steps:
            steps += 1
            
            # Get best action using minimax
            best_action = self._get_best_action(env, current, pending)
            
            if best_action is None:
                print(f"  No valid action at step {steps}")
                break
            
            # Check if blocked
            if env.is_blocked_by_adversary(best_action):
                # Try alternative routes
                alternatives = [n for n in env.graph.neighbors(current) 
                               if not env.is_blocked_by_adversary(n)]
                if alternatives:
                    best_action = alternatives[0]
                else:
                    print(f"  Blocked at step {steps}")
                    break
            
            # Execute move
            cost = env.graph[current][best_action]['weight']
            total_cost += cost
            path.append(best_action)
            current = best_action
            
            # Deliver package if at destination
            if current in pending:
                pending.remove(current)
            
            # Adversary responds (moves toward miner)
            if pending:
                env.move_adversary_toward(current)
        
        self.solution_path = path
        self.solution_cost = total_cost
        
        print(f"  Nodes expanded: {self.nodes_expanded}")
        print(f"  Packages delivered: {len(problem_instance.packages) - len(pending)}/{len(problem_instance.packages)}")
        
        if not pending:
            return path
        return path  # Return partial solution

    def _get_best_action(self, env, current, pending):
        """
        Select best action using Minimax with Alpha-Beta pruning.
        
        AIMA Figure 5.7: Alpha-Beta Search
        """
        self.nodes_expanded = 0
        
        neighbors = list(env.graph.neighbors(current))
        if not neighbors:
            return None
        
        best_action = None
        best_value = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        
        for action in neighbors:
            # Skip if blocked (unless all are blocked)
            if env.is_blocked_by_adversary(action):
                continue
            
            # Simulate action
            env_copy = env.clone()
            new_pending = [p for p in pending if p != action]
            
            # Adversary responds
            if new_pending:
                env_copy.move_adversary_toward(action)
            
            # Minimax evaluation (adversary's turn next)
            value = self._min_value(env_copy, action, new_pending, 
                                   self.max_depth - 1, alpha, beta)
            
            if value > best_value:
                best_value = value
                best_action = action
            
            alpha = max(alpha, value)
        
        # If all actions blocked, pick any neighbor
        if best_action is None and neighbors:
            best_action = neighbors[0]
        
        return best_action

    def _max_value(self, env, node, pending, depth, alpha, beta):
        """
        MAX's move (Miner) - AIMA Figure 5.7
        
        Returns utility value for MAX player.
        """
        self.nodes_expanded += 1
        
        # Terminal test
        if not pending:
            return 100  # Win
        if depth == 0:
            return self._evaluate(env, node, pending)
        
        value = float('-inf')
        
        for action in env.graph.neighbors(node):
            if env.is_blocked_by_adversary(action):
                continue
            
            env_copy = env.clone()
            new_pending = [p for p in pending if p != action]
            
            # Adversary responds
            if new_pending:
                env_copy.move_adversary_toward(action)
            
            value = max(value, self._min_value(env_copy, action, new_pending,
                                               depth - 1, alpha, beta))
            
            # Alpha-Beta pruning
            if value >= beta:
                return value  # Beta cutoff
            alpha = max(alpha, value)
        
        return value if value != float('-inf') else -100

    def _min_value(self, env, miner_node, pending, depth, alpha, beta):
        """
        MIN's move (Adversary) - AIMA Figure 5.7
        
        Returns utility value (adversary tries to minimize miner's score).
        """
        self.nodes_expanded += 1
        
        # Terminal test
        if not pending:
            return 100  # Miner wins anyway
        if depth == 0:
            return self._evaluate(env, miner_node, pending)
        
        value = float('inf')
        
        # Adversary moves
        adv_neighbors = env.get_adversary_neighbors()
        if not adv_neighbors:
            # Adversary can't move, skip to MAX
            return self._max_value(env, miner_node, pending, depth - 1, alpha, beta)
        
        for adv_action in adv_neighbors:
            env_copy = env.clone()
            env_copy.set_adversary_position(adv_action)
            
            value = min(value, self._max_value(env_copy, miner_node, pending,
                                               depth - 1, alpha, beta))
            
            # Alpha-Beta pruning
            if value <= alpha:
                return value  # Alpha cutoff
            beta = min(beta, value)
        
        return value

    def _evaluate(self, env, node, pending):
        """
        Evaluation function for non-terminal states.
        
        AIMA Section 5.4: "The evaluation function returns an estimate
        of the expected utility of the game from a given position."
        
        Features:
        - Number of remaining packages (negative)
        - Distance to nearest package (negative)
        - Distance from adversary (positive if far)
        """
        score = 0
        
        # Penalty for remaining packages
        score -= len(pending) * 20
        
        # Bonus for being close to packages
        if pending:
            coords = env.get_node_coordinates()
            node_pos = coords[node]
            
            min_dist = float('inf')
            for pkg in pending:
                pkg_pos = coords[pkg]
                from scipy.spatial import distance
                dist = distance.euclidean(node_pos, pkg_pos)
                min_dist = min(min_dist, dist)
            
            score -= min_dist * 0.1
        
        # Bonus for distance from adversary
        adv_pos = env.get_adversary_position()
        if adv_pos:
            coords = env.get_node_coordinates()
            from scipy.spatial import distance
            dist_to_adv = distance.euclidean(coords[node], coords[adv_pos])
            if dist_to_adv < 30:
                score -= (30 - dist_to_adv)  # Penalty for being close
        
        return score


if __name__ == "__main__":
    from environment import ProblemInstance
    
    print("Testing Minimax Miner (AIMA-based)...")
    env = ProblemInstance(num_nodes=15, random_seed=42, k_neighbors=5)
    
    print(f"Start: {env.get_start_node()}")
    print(f"Packages: {env.packages}")
    print(f"Adversary: {env.get_adversary_position()}")
    
    miner = NodeMinimax(max_depth=2)
    path = miner.solve(env)
    
    if path:
        print(f"\nSolution: {path}")
        print(f"Cost: {miner.solution_cost:.2f}")
    else:
        print("No solution found")
