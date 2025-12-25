"""
Main entry point for PoUW Consensus System.

This is a convenience wrapper around simulation.py for quick testing.
For full tournament functionality, use simulation.py directly.
"""

from environment import ProblemInstance
from simulation import MiningTournament, run_demo_tournament


def main():
    """
    Run the PoUW mining tournament.
    
    This simulates the blockchain mining competition where miners
    race to solve the logistics optimization problem.
    """
    print("=" * 60)
    print("PROOF-OF-USEFUL-WORK CONSENSUS SIMULATION")
    print("=" * 60)
    print()
    print("Miners compete to solve a 3-layer logistics problem:")
    print("  Layer 1: Base graph navigation (NP-hard)")
    print("  Layer 2: Stochastic traffic uncertainty")
    print("  Layer 3: Adversarial competition")
    print()
    
    # Run the tournament
    run_demo_tournament()


if __name__ == "__main__":
    main()
