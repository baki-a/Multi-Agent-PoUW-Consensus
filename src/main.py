import time
import random
import statistics
import os
import gob
import networkx as nx
from datetime import datetime
from environment import ProblemInstance
from miners import MinerRL
from miner_minimax import NodeMinimax
from miner_astar import NodeAStar
from miner_expectimax import NodeExpectimax


# Estructura blockchain, simulación muy breve
class Block:
    def __init__(self, index, timestamp, problem_name, solution_data, miner_name):
        self.index = index
        self.timestamp = timestamp
        self.problem_id = problem_name
        self.solution_path = solution_data['path']
        self.real_cost = solution_data['cost']
        self.miner_name = miner_name
        self.hash = hash(f"{index}{timestamp}{miner_name}{solution_data['real_cost']}{solution_data['path']}")


# El validador de bloques
class NetworkValidator:
    @staticmethod
    def validate_block(env, route):
        """
        Simula la ruta en el entorno real
        """

        if not route:
            return float('inf'), False

        real_cost = 0.0
        current_node = route[0]
        adversary_position = env.adversary_position
        caught_by_adversary = False

        for i in range(1, len(route)):
            next_node = route[i]

            # trafico real
            edge_data = env.graph[current_node][next_node]
            peso = edge_data['weight']
            
            if random.random() < edge_data.get('traffic_prob', 0.0):
                peso *= 2   # penalización de tráfico
            real_cost += peso
            

            # movimiento del adversario
            if adversary_position is not None:
                if adversary_position == current_node or adversary_position == next_node:
                    caught_by_adversary = True

                try:
                    path = nx.shortest_path(env.graph, adversary_position, next_node, weight='weight')
                    if len(path) > 1: 
                        adversary_position = path[1]
                except:
                    pass

                if adversary_position == next_node:
                    caught_by_adversary = True
                
            currrent_node = next_node

        return real_cost, caught_by_adversary

# Simulation motor
class PoUWConsensus:
    def __init__(self):
        self.chain = []
        self.miners = []
        self.stats = {}
        self.real_maps = self.load_real_maps()

    def load_real_maps(self):
        files = glob.glob("data/*.tsp") + glob.glob("*.tsp")
        if files:
            print(f"Mapas reales encontrados ({len(files)}): {[os.path.basename(f) for f in files]}")
        else:
            print("No se encontraron mapas reales")
        return files

    def register_miners(self, miners):
        self.miners = miners
        for miner in miners:
            self.stats[miner.name] = {'wins': 0, 'valid': 0, 'total_cost': 0}

    def start_minim_round(self, block_index):
        """
        Inicia una ronda de minería
        """
        if self.real_maps:
            map_file = self.real_maps[(block_index -1) % len(self.real_maps)]
            map_name = os.path.basename(map_file)
            env = ProblemInstance(tsplib_file=map_file, random_seed=42 + block_index, add_traffic=True)
        else:
            map_name = f"Random {block_index}"
            env = ProblemInstance(num_nodes=35, random_seed=42 + block_index, add_traffic=True)

        print(f"\nBloque #{block_index} || Mapa: {map_name}")
        print(f"Misión: {len(env.packages)} paquetes || Rival en {env.adversary_position}")   

        candidates = []

        for miner in self.miners:
            start = time.time()
            try:
                path = miner.solve(env)
            except:
                path = []
            duration = time.time() - start
            # empieza el consensus
            real_cost, is_valid = NetworkValidator.validate_block(env, path)
            
            candidates.append({'miner': miner, 'path': path, 'cost:' real_cost, 'valid': is_valid})
            string_result = "OK" if is_valid and real_cost < 3000 else "FAIL"
            print(f" - {miner.name:<15} || Tiempo: {duration:.3f}s || Coste: {real_cost:.1f} --> {string_result}")

        # sacamos el ganador
        valid_ones = [candidate for candidate in candidates if candidate['cost'] < 4000]
        if valid_ones:
            # gana el de menor coste real
            winner = min(valid_ones, key=lambda x: x['cost'])
            w_miner = winner['miner']

            self.chain.append(Block(block_index, datetime.now(), map_name, winner, w_miner.name))
            self.stats[w_miner.name]['wins'] += 1
            self.stats[w_miner.name]['valid'] += 1
            self.stats[w_miner.name]['total_cost'] += winner['cost']
            
            print(f"Ganador: {w_miner.name} || Coste: {winner['cost']:.1f}")
        else:
            print("Ningún candidato válido")

    def print_stats(self):
        print("\n " + "="*50)
        print("Resumen de estadísticas")
        print("="*50)

        print(f"{'Minero:'<20} || {'Bloques':<8} || {'Coste promedio':<12}")
        print("-"*50)

        for name, d in self.stats.items():
            average = d['total_cost'] / d['valid'] if d['valid'] > 0 else 0
            print(f"{name:<20} || {d['wins']:<8} || {average:<12.1f}")

def main():
    NUM_BLOCKS = 10
    sim = PoUWConsensus()
    
    miners = [
        NodeMinimax(),
        NodeAStar(),
        NodeExpectimax(),
        MinerRL()
    ]

    # entrenar RL
    print("Entrenando RL...")
    train_env = ProblemInstance(num_nodes=30, random_seed=100, add_traffic=True)
    miners[3].train(train_env, episodes=500)
    
    sim.register_miners(miners)

    # iniciar simulación
    for i in range(1, NUM_BLOCKS + 1):
        sim.start_minim_round(i)
    
    sim.print_stats()

if __name__ == "__main__":
    main()

            
        

        