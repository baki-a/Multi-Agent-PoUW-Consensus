import time
import random
import statistics
import os
import glob
import sys
from datetime import datetime
import networkx as nx

# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Local module imports - these are our core simulation components
from environment import ProblemInstance
from heuristic import Heuristic
from miner_astar import NodeAStar
from miner_expectimax import NodeExpectimax
from miner_minimax import NodeMinimax
from miner_rl import MinerRL

# ============================================================================
# CONFIGURATION CONSTANTS
# These control the "rescue" tie-breaker behavior when no miner hits threshold
# ============================================================================

# If the cost difference between two miners is less than 5%, we consider them "equal"
# and break the tie using solve time instead. This prevents a miner that takes 6 seconds
# to find a solution 1% better from beating one that solves in 0.1 seconds.
COST_SIMILARITY_THRESHOLD = 0.05

# Maximum number of nodes we allow in a map for the demo.
# Larger maps take too long and cause the presentation to drag.
MAX_NODES_FOR_DEMO = 90


class Block:
    """
    Represents a single "mined" block in our PoUW chain.
    Each block contains the solution to one VRP problem instance.
    """
    def __init__(self, block_idx, created_at, problem_name, solution_data, miner_name, threshold):
        self.index = block_idx
        self.timestamp = created_at
        self.problem_id = problem_name
        self.solution_path = solution_data['path']
        self.real_cost = solution_data['real_cost']
        self.threshold = threshold
        self.miner_name = miner_name
        # Simple hash for demo purposes - in a real blockchain this would be SHA-256
        self.hash = hash(f"{block_idx}{created_at}{miner_name}{solution_data['real_cost']}")


class NetworkValidator:
    """
    Validates solutions submitted by miners and calculates difficulty thresholds.
    Think of this as the "network consensus" that verifies work is actually useful.
    """
    
    @staticmethod
    def calculate_dynamic_threshold(env):
        """
        Calculate what counts as a "good enough" solution for this problem.
        We use a greedy baseline multiplied by a factor - this gives miners
        some slack while still requiring them to beat a naive approach.
        """
        start = env.get_start_node()
        coords = env.node_coordinates
        packages = list(env.packages)
        
        estimated_dist = 0
        curr = start
        temp_packages = packages.copy()
        
        # Greedy nearest-neighbor gives us a baseline cost
        while temp_packages:
            next_p = min(temp_packages, key=lambda p: Heuristic.euclidean_distance(curr, tuple([p]), coords))
            estimated_dist += Heuristic.euclidean_distance(curr, tuple([next_p]), coords)
            curr = next_p
            temp_packages.remove(next_p)
        
        # Multiply by 4x to set the threshold - we're generous here because
        # traffic and adversary can significantly inflate actual costs
        return max(estimated_dist * 4.0, 3000.0)

    @staticmethod
    def validate_block(env, route):
        """
        Simulate running the proposed route through the real environment.
        Returns (actual_cost, survived_without_capture).
        
        This is where stochastic elements (traffic) and adversarial elements
        (the chasing rival) affect the final outcome.
        """
        if not route:
            return float('inf'), False

        real_cost = 0.0
        current_node = route[0]
        adversary_pos = getattr(env, 'pos_adversario', getattr(env, 'adversary_pos', None))
        got_caught = False
        coords = env.node_coordinates

        for i in range(1, len(route)):
            next_node = route[i]
            
            # Traffic layer: with some probability, travel time doubles on this edge
            edge = env.graph[current_node][next_node]
            weight = edge['weight']
            if random.random() < edge.get('traffic_prob', 0.0):
                weight *= 2.0
            real_cost += weight

            # Adversary layer: the rival tries to catch us
            if adversary_pos is not None:
                # Check if we're at the same position as the rival
                if adversary_pos == current_node or adversary_pos == next_node:
                    got_caught = True
                
                # Calculate how close the rival is to decide if they should chase
                dist_to_rival = Heuristic.euclidean_distance(current_node, [adversary_pos], coords)
                
                # Limiting chase probability for demo - full pursuit makes it too hard
                should_chase = (dist_to_rival < 600) and (random.random() < 0.5)

                if should_chase:
                    try:
                        path = nx.shortest_path(env.graph, adversary_pos, next_node, weight='weight')
                        if len(path) > 1:
                            adversary_pos = path[1]
                    except:
                        pass
                else:
                    # Random wandering when not actively chasing
                    if random.random() < 0.5:
                        neighbors = list(env.graph.neighbors(adversary_pos))
                        if neighbors:
                            adversary_pos = random.choice(neighbors)
                
                if adversary_pos == next_node:
                    got_caught = True

            current_node = next_node
        
        # Heavy penalty for getting caught - effectively disqualifies the solution
        if got_caught:
            real_cost += 5000
            return real_cost, False
            
        return real_cost, True


class PoUWConsensus:
    """
    Main simulation controller. Manages the blockchain, registered miners,
    and orchestrates mining rounds.
    """
    
    def __init__(self):
        self.chain = []
        self.miners = []
        self.stats = {}
        self.round_history = []  # Track all rounds for the final report
        self.real_maps = self._load_real_maps()
        
    def _load_real_maps(self):
        """
        Load TSP problem files from the data directory.
        We filter out maps that are too large for the demo - 
        otherwise individual rounds can take minutes.
        """
        all_files = glob.glob("data/*.tsp") + glob.glob("*.tsp")
        valid_files = []
        
        print("")
        print("=" * 65)
        print("  SCANNING MAP FILES")
        print("=" * 65)
        
        for f in all_files:
            try:
                with open(f, 'r') as file:
                    for line in file:
                        if "DIMENSION" in line:
                            parts = line.split(':')
                            if len(parts) >= 2:
                                node_count = int(parts[1].strip())
                                
                                # Skip maps that are too large for demo purposes
                                if node_count < MAX_NODES_FOR_DEMO+10:
                                    print(f"  [OK] {os.path.basename(f):<25} ({node_count} nodes)")
                                    valid_files.append(f)
                                else:
                                    print(f"  [--] {os.path.basename(f):<25} ({node_count} nodes) - too large, skipping")
                            break
                            
            except Exception as e:
                print(f"  [!!] Error reading {f}: {e}")

        if not valid_files:
            print("  [!!] No suitable maps found. Will use procedurally generated ones.")
        
        print("=" * 65)
        print("")
        return valid_files

    def register_miners(self, miners):
        """Register the competing miners for the simulation."""
        self.miners = miners
        for m in miners:
            self.stats[m.name] = {'wins': 0, 'clean_wins': 0, 'total_cost': 0.0}
    
    def start_mining_round(self, block_idx):
        """
        Run a single mining round where all miners compete to solve the same VRP.
        The winner gets to "mine" this block.
        """
        dynamic_seed = int(time.time()) + block_idx * 55

        # Pick which map to use this round
        map_name = "Procedural"
        if self.real_maps:
            map_file = self.real_maps[(block_idx - 1) % len(self.real_maps)]
            map_name = os.path.basename(map_file)
            
        try:
            # Initialize the problem environment
            if self.real_maps:
                env = ProblemInstance(tsplib_file=map_file, random_seed=dynamic_seed, add_traffic=True)
            else:
                map_name = f"Procedural-{block_idx}"
                env = ProblemInstance(num_nodes=35, random_seed=dynamic_seed, add_traffic=True)

            # Safety check - sometimes map loading fails silently
            if not env.graph or len(env.graph.nodes) == 0:
                raise ValueError(f"Map {map_name} failed to load (empty graph).")

            threshold = NetworkValidator.calculate_dynamic_threshold(env)
            adversary_node = getattr(env, 'pos_adversario', getattr(env, 'adversary_pos', 'N/A'))

            # Round header
            print("")
            print("=" * 65)
            print(f"  ROUND #{block_idx} | {map_name} ({len(env.graph.nodes)} nodes) | Threshold: {threshold:.0f}")
            print("-" * 65)
            print(f"  Mission: {len(env.packages)} packages | Adversary at node: {adversary_node}")
            print("-" * 65)
            print(f"  {'MINER':<18} | {'TIME':>9} | {'COST':>10} | STATUS")
            print("-" * 65)

            # Let each miner attempt to solve the problem
            candidates = []
            for miner in self.miners:
                solve_start = time.time()
                path = []
                try:
                    path = miner.solve(env)
                except Exception as solve_error:
                    path = []
                solve_duration = time.time() - solve_start
                
                real_cost, survived = NetworkValidator.validate_block(env, path)
                passed_threshold = real_cost <= threshold
                is_valid_solution = survived and passed_threshold
                
                candidates.append({
                    'miner': miner,
                    'path': path,
                    'real_cost': real_cost,
                    'solve_time': solve_duration,
                    'valid': is_valid_solution,
                    'caught': not survived
                })
                
                # Format status with clear indicators
                if not survived:
                    status = "[X] CAUGHT"
                elif not passed_threshold:
                    status = "[X] INEFFICIENT"
                else:
                    status = "[OK]"
                    
                print(f"  {miner.name:<18} | {solve_duration:>8.3f}s | {real_cost:>10.1f} | {status}")

            print("-" * 65)

            # Select the winner using our refined logic
            valid_candidates = [c for c in candidates if c['valid']]
            is_rescue_round = False
            
            if not valid_candidates:
                # Nobody hit the threshold - this is a "rescue" scenario
                is_rescue_round = True
                survivors = [c for c in candidates if c['path']]
                not_caught = [c for c in survivors if not c['caught']]
                
                if not_caught:
                    valid_candidates = not_caught
                elif survivors:
                    print(f"  [!!] All miners caught! Selecting the fastest martyr...")
                    valid_candidates = survivors

            if valid_candidates:
                winner = self._pick_winner(valid_candidates, is_rescue_round)
                winning_miner = winner['miner']
                
                # Record the block
                self.chain.append(Block(
                    block_idx, datetime.now(), map_name, winner, 
                    winning_miner.name, threshold
                ))
                
                # Update stats
                self.stats[winning_miner.name]['wins'] += 1
                if not is_rescue_round:
                    self.stats[winning_miner.name]['clean_wins'] += 1
                self.stats[winning_miner.name]['total_cost'] += winner['real_cost']
                
                # Record for final report
                win_type = "CLEAN WIN" if not is_rescue_round else "RESCUE WIN"
                self.round_history.append({
                    'round': block_idx,
                    'map': map_name,
                    'winner': winning_miner.name,
                    'cost': winner['real_cost'],
                    'time': winner['solve_time'],
                    'type': win_type
                })
                
                print(f"  >> WINNER: {winning_miner.name} | Cost: {winner['real_cost']:.1f} | [{win_type}]")
            else:
                print(f"  [!!] ORPHAN BLOCK - No valid solution found")
                self.round_history.append({
                    'round': block_idx,
                    'map': map_name,
                    'winner': 'NONE',
                    'cost': 0,
                    'time': 0,
                    'type': 'ORPHAN'
                })
                
            print("=" * 65)

        except Exception as e:
            print("")
            print("=" * 65)
            print(f"  [!!] SKIPPING ROUND #{block_idx} ({map_name}): {e}")
            print("=" * 65)

    def _pick_winner(self, candidates, is_rescue):
        """
        Select the winning miner from valid candidates.
        
        In normal rounds: lowest cost wins.
        In rescue rounds: if the top two costs are within our threshold,
        we favor the faster solver to encourage efficiency.
        """
        sorted_by_cost = sorted(candidates, key=lambda x: x['real_cost'])
        
        if not is_rescue or len(sorted_by_cost) < 2:
            # Clean win - just pick the best cost
            return sorted_by_cost[0]
        
        # Rescue scenario - check if we should consider time
        first, second = sorted_by_cost[0], sorted_by_cost[1]
        
        # Calculate how different the costs are
        cost_gap = abs(first['real_cost'] - second['real_cost'])
        cost_baseline = max(first['real_cost'], 1.0)  # Avoid division by zero
        gap_ratio = cost_gap / cost_baseline
        
        if gap_ratio < COST_SIMILARITY_THRESHOLD:
            # Costs are essentially the same - pick the faster solver
            if second['solve_time'] < first['solve_time']:
                print(f"  -> Tie-breaker: {second['miner'].name} wins by speed ({second['solve_time']:.3f}s vs {first['solve_time']:.3f}s)")
                return second
        
        return first

    def print_final_stats(self):
        """Display the final leaderboard after all rounds."""
        print("")
        print("=" * 65)
        print("  FINAL STANDINGS")
        print("=" * 65)
        print(f"  {'MINER':<20} | {'BLOCKS':>8} | {'CLEAN':>6} | {'AVG COST':>10}")
        print("-" * 65)
        
        for name, data in sorted(self.stats.items(), key=lambda x: x[1]['wins'], reverse=True):
            total_wins = data['wins']
            clean_wins = data['clean_wins']
            
            # Avoid division by zero
            avg_cost = data['total_cost'] / max(total_wins, 1)
            
            print(f"  {name:<20} | {total_wins:>8} | {clean_wins:>6} | {avg_cost:>10.1f}")
        
        print("=" * 65)
        print("")
    
    def save_report(self, filename=None):
        """
        Save a summary report of the entire simulation run.
        Useful for comparing different algorithm configurations.
        """
        # Create reports directory if it doesn't exist
        os.makedirs("reports", exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"reports/run_report_{timestamp}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=" * 65 + "\n")
            f.write("  PoUW BLOCKCHAIN SIMULATION REPORT\n")
            f.write(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 65 + "\n\n")
            
            # Round-by-round breakdown
            f.write("ROUND RESULTS\n")
            f.write("-" * 65 + "\n")
            f.write(f"{'Round':<7} {'Map':<25} {'Winner':<18} {'Cost':>10} {'Type':<12}\n")
            f.write("-" * 65 + "\n")
            
            for entry in self.round_history:
                f.write(f"{entry['round']:<7} {entry['map']:<25} {entry['winner']:<18} {entry['cost']:>10.1f} {entry['type']:<12}\n")
            
            f.write("\n")
            
            # Final standings
            f.write("FINAL STANDINGS\n")
            f.write("-" * 65 + "\n")
            f.write(f"{'Miner':<20} {'Blocks':>10} {'Clean Wins':>12} {'Avg Cost':>12}\n")
            f.write("-" * 65 + "\n")
            
            for name, data in sorted(self.stats.items(), key=lambda x: x[1]['wins'], reverse=True):
                total_wins = data['wins']
                clean_wins = data['clean_wins']
                avg_cost = data['total_cost'] / max(total_wins, 1)
                f.write(f"{name:<20} {total_wins:>10} {clean_wins:>12} {avg_cost:>12.1f}\n")
            
            f.write("=" * 65 + "\n")
        
        print(f"  [>>] Report saved to: {filename}")


def main():
    """Main entry point for the PoUW blockchain simulation."""
    sim = PoUWConsensus()
    
    # Initialize our four competing miners:
    # - A* (optimal pathfinding, no uncertainty modeling)
    # - Expectimax (handles stochastic traffic)
    # - Minimax (handles adversarial rival)
    # - RL Supernode (learned behavior, fast inference)
    
    miners = [
        NodeAStar(),
        NodeExpectimax(),
        # Limiting Minimax time to 0.3s per turn so the demo doesn't stall.
        # In production you'd want this higher, but for presentation purposes
        # we need rounds to complete in reasonable time.
        NodeMinimax(time_limit=0.3),
        MinerRL()
    ]

    # Train the RL agent before competition begins
    # 3000 episodes is a balance between training quality and startup time
    print("")
    print("=" * 65)
    print("  TRAINING RL AGENT (3000 episodes)")
    print("=" * 65)
    training_env = ProblemInstance(num_nodes=35, random_seed=999, add_traffic=True)
    miners[3].train(training_env, episodes=3000)
    print("  [OK] Training complete")
    print("=" * 65)

    sim.register_miners(miners)
    
    # Run one round per available map (minimum 5 rounds)
    num_maps = len(sim.real_maps)
    total_rounds = max(num_maps, 5)
    
    print("")
    print("=" * 65)
    print(f"  STARTING COMPETITION: {total_rounds} rounds")
    print("=" * 65)
    
    for round_num in range(1, total_rounds + 1):
        sim.start_mining_round(round_num)
    
    sim.print_final_stats()
    sim.save_report()


if __name__ == "__main__":
    main()