"""
Simulation Framework - Tournament Runner for PoUW Mining Competition

Runs all miner types in parallel (simulated) and determines the winner
based on solution quality and computation time.
"""

import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from environment import ProblemInstance
from miners import MinerAbstract
from miner_astar import NodeAStar
from miner_expectimax import NodeExpectimax
from miner_minimax import NodeMinimax

# Import RL miner conditionally
try:
    from miner_rl import SupernodeRL, SB3_AVAILABLE
except ImportError:
    SB3_AVAILABLE = False
    SupernodeRL = None


@dataclass
class MiningResult:
    """Result of a mining attempt."""
    miner_name: str
    path: Optional[List[int]]
    cost: float
    time_seconds: float
    packages_delivered: int
    total_packages: int
    success: bool
    
    @property
    def quality_score(self) -> float:
        """Calculate quality score: higher is better."""
        if not self.success:
            return 0.0
        # Score = packages delivered / cost, normalized
        return (self.packages_delivered / max(self.cost, 1)) * 1000


class MiningTournament:
    """
    Tournament runner for PoUW mining competition.
    Simulates head-to-head competition between different miner types.
    """
    
    def __init__(self, quality_threshold: float = 0.8, time_limit: float = 60.0):
        """
        Initialize tournament.
        
        Args:
            quality_threshold: Minimum quality ratio to "win" (packages delivered %)
            time_limit: Maximum time per miner in seconds
        """
        self.quality_threshold = quality_threshold
        self.time_limit = time_limit
        self.results: List[MiningResult] = []
    
    def create_miners(self, include_rl: bool = True) -> List[MinerAbstract]:
        """Create all miner instances for competition."""
        miners = [
            NodeAStar(),
            NodeExpectimax(),
            NodeMinimax(max_depth=3),
        ]
        
        if include_rl and SB3_AVAILABLE and SupernodeRL:
            # Check for pre-trained model first
            import os
            model_path = "models/ppo_vrp.zip"
            if os.path.exists(model_path):
                print(f"Loading pre-trained RL model from {model_path}")
                rl_miner = SupernodeRL()
                rl_miner.load(model_path.replace('.zip', ''))
                miners.append(rl_miner)
            else:
                print("No pre-trained model found. Training from scratch (slow)...")
                print("  Tip: Run 'python train_rl.py' first for better results")
                miners.append(SupernodeRL(training_timesteps=10000))
        
        return miners
    
    def run_miner(self, miner: MinerAbstract, problem: ProblemInstance) -> MiningResult:
        """
        Run a single miner and record results.
        
        Args:
            miner: The miner to run
            problem: Problem instance (will be cloned)
            
        Returns:
            MiningResult with performance data
        """
        # Clone environment for fair competition
        env = problem.clone()
        total_packages = len(env.packages)
        
        start_time = time.time()
        
        try:
            path = miner.solve(env)
            elapsed = time.time() - start_time
            
            if path:
                # Count delivered packages
                delivered = sum(1 for pkg in problem.packages if pkg in path)
                success = delivered >= total_packages * self.quality_threshold
                
                return MiningResult(
                    miner_name=miner.name,
                    path=path,
                    cost=miner.solution_cost,
                    time_seconds=elapsed,
                    packages_delivered=delivered,
                    total_packages=total_packages,
                    success=success
                )
            else:
                return MiningResult(
                    miner_name=miner.name,
                    path=None,
                    cost=float('inf'),
                    time_seconds=elapsed,
                    packages_delivered=0,
                    total_packages=total_packages,
                    success=False
                )
                
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"Error in {miner.name}: {e}")
            return MiningResult(
                miner_name=miner.name,
                path=None,
                cost=float('inf'),
                time_seconds=elapsed,
                packages_delivered=0,
                total_packages=total_packages,
                success=False
            )
    
    def run_tournament(self, problem: ProblemInstance, 
                       parallel: bool = True,
                       include_rl: bool = False) -> Dict[str, Any]:
        """
        Run full tournament with all miners.
        
        Args:
            problem: Problem instance for competition
            parallel: If True, run miners in parallel threads
            include_rl: If True, include RL miner (slower due to training)
            
        Returns:
            Dictionary with tournament results
        """
        miners = self.create_miners(include_rl=include_rl)
        self.results = []
        
        print("=" * 60)
        print("POUW MINING TOURNAMENT")
        print("=" * 60)
        print(f"Nodes: {problem.graph.number_of_nodes()}")
        print(f"Packages: {len(problem.packages)}")
        print(f"Quality threshold: {self.quality_threshold * 100}%")
        print(f"Miners: {[m.name for m in miners]}")
        print("=" * 60)
        
        if parallel:
            # Run miners in parallel
            with ThreadPoolExecutor(max_workers=len(miners)) as executor:
                futures = {
                    executor.submit(self.run_miner, miner, problem): miner 
                    for miner in miners
                }
                
                for future in as_completed(futures):
                    result = future.result()
                    self.results.append(result)
                    self._print_result(result)
        else:
            # Run sequentially
            for miner in miners:
                result = self.run_miner(miner, problem)
                self.results.append(result)
                self._print_result(result)
        
        # Determine winner
        return self._determine_winner()
    
    def _print_result(self, result: MiningResult):
        """Print individual miner result."""
        status = "[SUCCESS]" if result.success else "[FAILED]"
        print(f"\n{result.miner_name}: {status}")
        print(f"  Packages: {result.packages_delivered}/{result.total_packages}")
        print(f"  Cost: {result.cost:.2f}")
        print(f"  Time: {result.time_seconds:.3f}s")
        print(f"  Quality Score: {result.quality_score:.2f}")
    
    def _determine_winner(self) -> Dict[str, Any]:
        """
        Determine tournament winner.
        Winner = First to cross quality threshold with best score.
        """
        print("\n" + "=" * 60)
        print("TOURNAMENT RESULTS")
        print("=" * 60)
        
        # Filter successful miners
        successful = [r for r in self.results if r.success]
        
        if not successful:
            print("No miner succeeded! No winner.")
            return {
                'winner': None,
                'all_results': self.results,
                'message': "No miner crossed quality threshold"
            }
        
        # Sort by: 1) Quality score (desc), 2) Time (asc)
        successful.sort(key=lambda r: (-r.quality_score, r.time_seconds))
        
        winner = successful[0]
        
        print(f"\n*** WINNER: {winner.miner_name} ***")
        print(f"   Quality Score: {winner.quality_score:.2f}")
        print(f"   Cost: {winner.cost:.2f}")
        print(f"   Time: {winner.time_seconds:.3f}s")
        
        print("\nRANKINGS:")
        for i, result in enumerate(successful, 1):
            print(f"  {i}. {result.miner_name} (Score: {result.quality_score:.2f})")
        
        return {
            'winner': winner.miner_name,
            'winner_result': winner,
            'rankings': successful,
            'all_results': self.results
        }


def run_demo_tournament():
    """Run a demo tournament for testing."""
    print("\n" + "=" * 60)
    print("DEMO: PoUW Mining Tournament")
    print("=" * 60)
    
    # Check RL availability
    if SB3_AVAILABLE:
        print("RL Miner: AVAILABLE (stable-baselines3 installed)")
    else:
        print("RL Miner: NOT AVAILABLE")
        print("  To enable: pip install stable-baselines3 gymnasium")
    print()
    
    # Create problem with sparse graph (k=5 neighbors)
    problem = ProblemInstance(
        num_nodes=20,
        random_seed=42,
        k_neighbors=5
    )
    
    # Run tournament - include RL if available
    tournament = MiningTournament(quality_threshold=0.8)
    results = tournament.run_tournament(
        problem, 
        parallel=False, 
        include_rl=SB3_AVAILABLE  # Include RL if stable-baselines3 is installed
    )
    
    return results


if __name__ == "__main__":
    run_demo_tournament()
