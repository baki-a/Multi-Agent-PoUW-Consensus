# Multi-Agent PoUW Consensus

A blockchain simulation implementing **Proof-of-Useful-Work (PoUW)** through the Vehicle Routing Problem (VRP), featuring multiple AI agents competing to solve real-world logistics problems.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Research-orange.svg)

---

## Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Agents/Miners](#-agentsminers)
- [Simulation Results](#-simulation-results)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)

---

## Overview

This project reimagines blockchain consensus by replacing traditional Proof-of-Work (solving hash puzzles) with **Proof-of-Useful-Work**: solving real-world optimization problems that have actual business value.

In our simulation, multiple AI agents ("miners") **compete to find optimal delivery routes** on real maps. The miner that produces the best solution wins the block. This creates a system where:

-  Computational power is directed toward solving **real problems**
-  Multiple AI strategies compete for block production
-  Uses real-world TSPLIB benchmark datasets

---

##  Features

- **Multi-layered Environment**: Base map + Traffic uncertainty + Adversarial agents
- **4 AI Mining Strategies**: A*, Expectimax, Minimax with Alpha-Beta pruning, Q-Learning RL
- **Real TSPLIB Maps**: Berlin52, att48, eil51, and 20+ more
- **Dynamic Difficulty**: Threshold adjusts based on map complexity
- **Stochastic Traffic**: 20% of roads have random congestion probability
- **Adversarial Layer**: Rival agent that pursues and penalizes miners

---

## Architecture

The simulation environment (`environment.py`) uses a **3-layer architecture**:

```
┌─────────────────────────────────────────────────────────┐
│  Layer 3: ADVERSARIAL                                   │
│  • Rival agent placement                                │
│  • Pursuit mechanics (85% accuracy within 900 units)    │
│  • Capture penalty: +5000 cost                          │
├─────────────────────────────────────────────────────────┤
│  Layer 2: STOCHASTIC (Traffic)                          │
│  • 20% of edges have traffic probability                │
│  • Traffic prob: 10% - 50%                              │
│  • Effect: Cost doubles if traffic triggered            │
├─────────────────────────────────────────────────────────┤
│  Layer 1: BASE (NP-Hard)                                │
│  • Complete graph with Euclidean weights                │
│  • TSPLIB format support (.tsp files)                   │
│  • Random graph generation option                       │
└─────────────────────────────────────────────────────────┘
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/Multi-Agent-PoUW-Consensus.git
cd Multi-Agent-PoUW-Consensus

# Install dependencies
pip install networkx numpy scipy

# Run the simulation
cd src
python main.py
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `networkx` | Graph representation and algorithms |
| `numpy` | Numerical operations and RL training |
| `scipy` | Euclidean distance calculations |

---

## Usage

### Running the Full Simulation

```bash
cd src
python main.py
```

This will:
1. Scan and validate available TSPLIB maps in `data/`
2. Train the RL agent for 3000 episodes
3. Run mining rounds on each valid map
4. Display final standings and generate a report

### Testing Individual Components

```bash
# Test the environment
python environment.py

# Output: Graph inspection with traffic zones
```

### Sample Output

```
BLOCK #1 | berlin52.tsp (52 nodes) | Threshold: 8500
Mission: 5 packages | Rival at: 23
---------------------------------------------------------------------------
MINER              |    TIME |  REAL COST | STATUS
---------------------------------------------------------------------------
Miner A*           | 0.023s  |     6265.6 | OK
NodeExpectimax     | 0.045s  |     5932.1 | OK
NodeMinimax        | 0.512s  |     7105.3 | OK
Supernode-RL       | 0.001s  |     5516.7 | OK
---------------------------------------------------------------------------
   WINNER: Supernode-RL (Cost: 5516.7) [CLEAN WIN]
```

---

## Agents/Miners

### 1. Miner A* (`miner_astar.py`)

Classic informed search algorithm.

- **Strategy**: Finds shortest path ignoring traffic and adversary
- **Heuristic**: Euclidean distance to closest pending package
- **Strengths**: Fast, optimal for static graphs
- **Weaknesses**: Ignores stochastic and adversarial layers

### 2. NodeExpectimax (`miner_expectimax.py`)

A* variant with expected cost calculation.

- **Strategy**: Considers traffic probability in edge costs
- **Formula**: `expected_cost = base + (base × traffic_prob × penalty)`
- **Strengths**: Better performance in high-traffic zones
- **Weaknesses**: Ignores adversary

### 3. NodeMinimax (`miner_minimax.py`)

Adversarial search with Alpha-Beta pruning.

- **Strategy**: Assumes adversary plays optimally against us
- **Features**: 
  - Iterative deepening (depth 1-4)
  - Time limit: 0.5s per move
  - Smart fallback to greedy if timeout
- **Strengths**: Best against aggressive adversaries
- **Weaknesses**: High computational cost, limited depth

### 4. Supernode-RL (`miner_rl.py`)

Q-Learning agent with abstract action space.

- **States**: (Danger Level, Traffic Level) - 9 possible states
- **Actions**:
  - `0`: Greedy (go to closest package)
  - `1`: Smart (avoid traffic)
  - `2`: Defensive (flee from adversary)
- **Training**: 3000 episodes with epsilon decay
- **Strengths**: Adapts to environment, best average cost
- **Weaknesses**: Requires pre-training

---

## Simulation Results

Sample results from a 22-round simulation:

| Miner | Blocks Won | Clean Wins | Avg Cost |
|-------|------------|------------|----------|
| Miner A* | 4 | 1 | 7728.8 |
| Supernode-RL | 4 | 0 | 5147.8 |
| NodeExpectimax | 3 | 1 | 6597.6 |
| NodeMinimax | 0 | 0 | - |

### Key Insights

- **A\*** wins on speed but has highest average cost
- **Supernode-RL** achieves best cost efficiency (33% better than A*)
- **Expectimax** offers good balance of speed and quality
- **Minimax** struggles with time constraints

---

##  Project Structure

```
Multi-Agent-PoUW-Consensus/
├── 📂 data/                    # TSPLIB benchmark files
│   ├── berlin52.tsp
│   ├── att48.tsp
│   └── ... (113 files)
├── 📂 src/
│   ├── main.py                 # Simulation orchestrator
│   ├── environment.py          # 3-layer problem environment
│   ├── miners.py               # Abstract miner base class
│   ├── miner_astar.py          # A* implementation
│   ├── miner_expectimax.py     # Expectimax implementation
│   ├── miner_minimax.py        # Minimax with Alpha-Beta
│   ├── miner_rl.py             # Q-Learning agent
│   └── heuristic.py            # Heuristic functions
├── 📂 reports/                 # Generated simulation reports
├── 📂 docs/                    # Documentation
└── README.md
```

---

##  Technical Details

### Consensus Mechanism

1. **Challenge**: Generate VRP instance from map + random seed
2. **Threshold**: Dynamic, calculated as `4 × Greedy_Solution`
3. **Validation**: Simulate route with real traffic and adversary
4. **Winner Selection**: 
   - Primary: Lowest real cost under threshold
   - Tiebreaker: Fastest solve time (within 5% cost similarity)
5. **Rescue Protocol**: If all fail threshold, best attempt wins

### Heuristics (`heuristic.py`)

```python
# Euclidean distance to closest package
def euclidean_distance(current, pending, coords) -> float

# Minimax evaluation function
def evaluate_minimax(current, pending, adversary, coords, cost, total) -> float
# Considers: packages delivered (+1000 each), distance penalty, 
#            adversary proximity (-5000 if too close), total cost
```

---

##  Contributing

Contributions are welcome! Some ideas:

- [ ] Implement Monte Carlo Tree Search (MCTS) miner
- [ ] Add Deep Q-Network (DQN) agent
- [ ] Create visualization dashboard
- [ ] Implement actual blockchain with blocks and chain validation
- [ ] Add more TSPLIB instances and benchmark comparisons

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [TSPLIB](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/) for benchmark datasets
- *Artificial Intelligence: A Modern Approach* (Russell & Norvig) for algorithm foundations
- The blockchain and optimization research communities

---

<p align="center">
  Made with ❤️ for AI and Blockchain research
</p>
