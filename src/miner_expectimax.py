"""
Expectimax Miner - Handles Layer 1 (Base) + Layer 2 (Stochastic Traffic)

Uses Expectimax search to account for traffic uncertainty.
Chance nodes represent probabilistic traffic conditions.
"""

import heapq
from miners import MinerAbstract
from heuristic import Heuristic


class NodeExpectimax(MinerAbstract):
    def __init__(self, max_depth=3, num_samples=10):
        """
        Initialize Expectimax miner.
        
        Args:
            max_depth: Maximum search depth for expectimax lookahead
            num_samples: Number of Monte Carlo samples for chance nodes
        """
        super().__init__(name="Miner Expectimax")
        self.max_depth = max_depth
        self.num_samples = num_samples

    def solve(self, problem_instance):
        """
        Solve VRP using A* with expected costs for the heuristic,
        accounting for stochastic traffic.
        """
        start_node = problem_instance.get_start_node()
        coordinates = problem_instance.get_node_coordinates()
        
        # Initial state: (current_node, pending_packages)
        package_initial = tuple(sorted(problem_instance.packages))
        
        # Priority queue: (f_score, g_score, current_node, path, pending_packages)
        priority_queue = []
        h_start = self._expected_heuristic(start_node, package_initial, problem_instance)
        heapq.heappush(priority_queue, (h_start, 0, start_node, [start_node], package_initial))
        
        # Visited states dictionary
        visited = {}
        
        print("Resolviendo problema con Expectimax (Layer 1+2)...")
        
        while priority_queue:
            f, g, current_state, path, pending = heapq.heappop(priority_queue)
            
            # Goal check: all packages delivered
            if not pending:
                self.solution_path = path
                self.solution_cost = g
                return path
            
            # State key for visited check
            visited_state = (current_state, pending)
            
            if visited_state in visited and visited[visited_state] <= g:
                continue
            
            visited[visited_state] = g
            
            # Expand using expected costs
            for neighbor in problem_instance.graph.neighbors(current_state):
                # Use EXPECTED cost instead of base cost
                expected_cost = problem_instance.get_expected_cost(current_state, neighbor)
                new_g = g + expected_cost
                
                # Update pending packages
                new_pending_list = list(pending)
                if neighbor in new_pending_list:
                    new_pending_list.remove(neighbor)
                new_pending_tuple = tuple(sorted(new_pending_list))
                
                # Expected heuristic
                h = self._expected_heuristic(neighbor, new_pending_tuple, problem_instance)
                new_f = new_g + h
                
                heapq.heappush(priority_queue, 
                              (new_f, new_g, neighbor, path + [neighbor], new_pending_tuple))
        
        return None

    def _expected_heuristic(self, current_node, pending_packages, problem_instance):
        """
        Calculate expected heuristic considering traffic probabilities.
        Uses minimum expected distance to nearest pending package.
        """
        if not pending_packages:
            return 0.0
        
        coordinates = problem_instance.get_node_coordinates()
        current_pos = coordinates[current_node]
        
        min_expected_dist = float('inf')
        
        for package in pending_packages:
            package_pos = coordinates[package]
            # Base euclidean distance
            from scipy.spatial import distance
            base_dist = distance.euclidean(current_pos, package_pos)
            
            # Estimate expected overhead from traffic (average 10% increase)
            expected_dist = base_dist * 1.1
            
            if expected_dist < min_expected_dist:
                min_expected_dist = expected_dist
        
        return min_expected_dist

    def expectimax_lookahead(self, problem_instance, current_node, pending, depth, is_chance):
        """
        Expectimax lookahead for decision making.
        
        Args:
            problem_instance: The problem environment
            current_node: Current position
            pending: Remaining packages
            depth: Remaining search depth
            is_chance: True if this is a chance node
            
        Returns:
            Expected value of being in this state
        """
        # Terminal conditions
        if not pending:
            return 0  # Goal reached
        
        if depth == 0:
            return -self._expected_heuristic(current_node, pending, problem_instance)
        
        if is_chance:
            # Chance node: average over traffic outcomes
            total_value = 0
            neighbors = list(problem_instance.graph.neighbors(current_node))
            
            if not neighbors:
                return float('-inf')
            
            for neighbor in neighbors:
                # Sample traffic outcome
                cost = problem_instance.get_expected_cost(current_node, neighbor)
                
                new_pending = tuple(p for p in pending if p != neighbor)
                future_value = self.expectimax_lookahead(
                    problem_instance, neighbor, new_pending, depth - 1, False
                )
                total_value += (-cost + future_value) / len(neighbors)
            
            return total_value
        else:
            # Max node: choose best action
            best_value = float('-inf')
            
            for neighbor in problem_instance.graph.neighbors(current_node):
                value = self.expectimax_lookahead(
                    problem_instance, neighbor, pending, depth, True
                )
                best_value = max(best_value, value)
            
            return best_value


if __name__ == "__main__":
    from environment import ProblemInstance
    
    print("Testing Expectimax Miner...")
    env = ProblemInstance(num_nodes=15, random_seed=42, k_neighbors=5)
    
    miner = NodeExpectimax()
    path = miner.solve(env)
    
    if path:
        print(f"Solution found: {path}")
        print(f"Expected cost: {miner.solution_cost:.2f}")
    else:
        print("No solution found")
