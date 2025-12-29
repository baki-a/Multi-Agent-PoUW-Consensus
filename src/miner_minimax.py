import heapq
from miners import MinerAbstract
from heuristic import Heuristic

class NodeMinimax(MinerAbstract):
    """
    Minimax miner that considers adversarial competition.
    Uses Alpha-Beta pruning for efficiency.
    """
    
    def __init__(self, max_depth=3):
        super().__init__(name="Miner Minimax")
        self.max_depth = max_depth
        self.nodes_expanded = 0
        self.problem = None
        self.coordinates = None
    
    def solve(self, problem_instance):
        """
        Solve VRP using Minimax with Alpha-Beta pruning.
        
        Strategy: Use greedy approach with minimax lookahead.
        At each step, evaluate possible moves using minimax
        to account for adversary's best response.
        """
        print(f"Resolviendo problema con Minimax (max_depth={self.max_depth})...")
        
        self.problem = problem_instance
        self.coordinates = problem_instance.get_node_coordinates()
        self.nodes_expanded = 0
        
        start_node = problem_instance.get_start_node()
        pending = set(problem_instance.packages)
        adversary_pos = problem_instance.pos_adversario
        
        path = [start_node]
        current = start_node
        total_cost = 0
        
        # Greedy loop with minimax lookahead
        while pending:
            # Get best next move using minimax
            best_move, best_value = self._get_best_move(
                current, pending, adversary_pos, total_cost
            )
            
            if best_move is None:
                # No valid move found (blocked)
                print(f"  [BLOCKED] No valid moves from {current}")
                break
            
            # Execute the move
            edge_cost = self.problem.graph[current][best_move]['weight']
            total_cost += edge_cost
            path.append(best_move)
            
            # Deliver package if present
            if best_move in pending:
                pending.remove(best_move)
            
            # Simulate adversary response (moves toward miner)
            adversary_pos = self._move_adversary_toward(adversary_pos, best_move)
            
            current = best_move
        
        self.solution_path = path
        self.solution_cost = total_cost
        
        print(f"  Nodes expanded: {self.nodes_expanded}")
        print(f"  Packages delivered: {len(problem_instance.packages) - len(pending)}/{len(problem_instance.packages)}")
        
        return path
    
    def _get_best_move(self, current, pending, adversary_pos, current_cost):
        """
        Use Minimax with Alpha-Beta to find the best next move.
        
        Based on AIMA Fig 5.7: ALPHA-BETA-SEARCH
        """
        best_move = None
        best_value = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        
        # Get all possible moves (neighbors not blocked by adversary)
        neighbors = list(self.problem.graph.neighbors(current))
        
        for move in neighbors:
            # Skip if blocked by adversary
            if move == adversary_pos:
                continue
            
            # Simulate the move
            new_pending = pending.copy()
            if move in new_pending:
                new_pending.remove(move)
            
            edge_cost = self.problem.graph[current][move]['weight']
            new_cost = current_cost + edge_cost
            
            # Adversary responds (MIN's turn)
            new_adversary = self._move_adversary_toward(adversary_pos, move)
            
            # Evaluate using minimax
            value = self._min_value(
                move, new_pending, new_adversary, new_cost,
                alpha, beta, self.max_depth - 1
            )
            
            if value > best_value:
                best_value = value
                best_move = move
            
            alpha = max(alpha, best_value)
        
        return best_move, best_value
    
    def _max_value(self, current, pending, adversary_pos, cost, alpha, beta, depth):
        """
        MAX player's turn (Miner).
        
        Based on AIMA Fig 5.7: MAX-VALUE function
        
        function MAX-VALUE(state, α, β) returns a utility value
            if TERMINAL-TEST(state) then return UTILITY(state)
            v ← -∞
            for each a in ACTIONS(state) do
                v ← MAX(v, MIN-VALUE(RESULT(state, a), α, β))
                if v ≥ β then return v  # β cutoff
                α ← MAX(α, v)
            return v
        """
        self.nodes_expanded += 1
        
        # Terminal test
        if depth <= 0 or not pending:
            return self._evaluate(current, pending, adversary_pos, cost)
        
        v = float('-inf')
        neighbors = list(self.problem.graph.neighbors(current))
        
        for move in neighbors:
            # Skip blocked moves
            if move == adversary_pos:
                continue
            
            new_pending = pending.copy()
            if move in new_pending:
                new_pending.remove(move)
            
            edge_cost = self.problem.graph[current][move]['weight']
            new_cost = cost + edge_cost
            new_adversary = self._move_adversary_toward(adversary_pos, move)
            
            v = max(v, self._min_value(
                move, new_pending, new_adversary, new_cost,
                alpha, beta, depth - 1
            ))
            
            # Beta cutoff (pruning)
            if v >= beta:
                return v
            
            alpha = max(alpha, v)
        
        return v
    
    def _min_value(self, current, pending, adversary_pos, cost, alpha, beta, depth):
        """
        MIN player's turn (Adversary).
        
        Based on AIMA Fig 5.7: MIN-VALUE function
        
        function MIN-VALUE(state, α, β) returns a utility value
            if TERMINAL-TEST(state) then return UTILITY(state)
            v ← +∞
            for each a in ACTIONS(state) do
                v ← MIN(v, MAX-VALUE(RESULT(state, a), α, β))
                if v ≤ α then return v  # α cutoff
                β ← MIN(β, v)
            return v
        """
        self.nodes_expanded += 1
        
        # Terminal test
        if depth <= 0 or not pending:
            return self._evaluate(current, pending, adversary_pos, cost)
        
        v = float('inf')
        
        # Adversary's possible moves (neighbors of adversary position)
        adv_neighbors = list(self.problem.graph.neighbors(adversary_pos))
        
        for adv_move in adv_neighbors:
            # Adversary moves toward miner (greedy strategy)
            new_adversary = adv_move
            
            v = min(v, self._max_value(
                current, pending, new_adversary, cost,
                alpha, beta, depth - 1
            ))
            
            # Alpha cutoff (pruning)
            if v <= alpha:
                return v
            
            beta = min(beta, v)
        
        return v
    
    def _evaluate(self, current, pending, adversary_pos, cost):
        """
        Evaluation function for non-terminal states.
        
        Higher values are better for MAX (miner).
        
        Factors:
        - Packages delivered (+points)
        - Distance to nearest package (-points)
        - Distance from adversary (+points for safety)
        - Accumulated cost (-points)
        """
        score = 0.0
        
        # Reward delivered packages (major factor)
        delivered = len(self.problem.packages) - len(pending)
        score += delivered * 1000
        
        # Penalize remaining distance to packages
        if pending:
            nearest_dist = min(
                Heuristic.euclidean_distance(current, tuple([p]), self.coordinates)
                for p in pending
            )
            score -= nearest_dist
        
        # Reward distance from adversary (safety)
        if adversary_pos is not None and adversary_pos in self.coordinates:
            adv_dist = Heuristic.euclidean_distance(
                current, tuple([adversary_pos]), self.coordinates
            )
            score += adv_dist * 0.5  # Safety bonus
        
        # Penalize accumulated cost
        score -= cost * 0.1
        
        return score
    
    def _move_adversary_toward(self, adversary_pos, target):
        """
        Move adversary one step toward the target using shortest path.
        
        This simulates the adversary's greedy pursuit strategy.
        """
        if adversary_pos is None or adversary_pos == target:
            return adversary_pos
        
        try:
            import networkx as nx
            path = nx.shortest_path(
                self.problem.graph, adversary_pos, target, weight='weight'
            )
            if len(path) > 1:
                return path[1]  # Next step toward target
        except:
            pass
        
        return adversary_pos


# Quick test
if __name__ == "__main__":
    from environment import ProblemInstance
    
    print("Testing Minimax Miner...")
    env = ProblemInstance(num_nodes=15, random_seed=42)
    
    miner = NodeMinimax(max_depth=3)
    path = miner.solve(env)
    
    print(f"\nSolution path: {path}")
    print(f"Total cost: {miner.solution_cost:.2f}")
    print(f"Packages: {env.packages}")
