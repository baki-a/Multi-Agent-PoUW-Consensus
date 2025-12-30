import time
import random
import statistics
import os
import glob
from datetime import datetime
import networkx as nx

# Importa tus modulos
from environment import ProblemInstance
from heuristic import Heuristic
from miner_astar import NodeAStar
from miner_expectimax import NodeExpectimax
from miner_minimax import NodeMinimax
from miner_rl import MinerRL 

class Block:
    def __init__(self, index, timestamp, problem_name, solution_data, miner_name, threshold):
        self.index = index
        self.timestamp = timestamp
        self.problem_id = problem_name 
        self.solution_path = solution_data['path']
        # AQUI estaba el error, ahora solution_data si tendra 'real_cost'
        self.real_cost = solution_data['real_cost'] 
        self.threshold = threshold
        self.miner_name = miner_name
        self.hash = hash(f"{index}{timestamp}{miner_name}{solution_data['real_cost']}")

class NetworkValidator:
    @staticmethod
    def calculate_dynamic_threshold(env):
        start = env.get_start_node()
        coords = env.node_coordinates
        packages = list(env.packages)
        
        estimated_dist = 0
        curr = start
        temp_packages = packages.copy()
        
        # Heuristica Greedy para base
        while temp_packages:
            next_p = min(temp_packages, key=lambda p: Heuristic.euclidean_distance(curr, tuple([p]), coords))
            estimated_dist += Heuristic.euclidean_distance(curr, tuple([next_p]), coords)
            curr = next_p
            temp_packages.remove(next_p)
            
        # Multiplicador x3.5. Muy permisivo para que entren soluciones seguras aunque sean largas
        return max(estimated_dist * 3.5, 2000.0)

    @staticmethod
    def validate_block(env, route):
        if not route: return float('inf'), False

        real_cost = 0.0
        current_node = route[0]
        adversary_pos = getattr(env, 'pos_adversario', getattr(env, 'adversary_pos', None))
        caught = False

        for i in range(1, len(route)):
            next_node = route[i]
            
            # 1. Trafico
            edge = env.graph[current_node][next_node]
            weight = edge['weight']
            if random.random() < edge.get('traffic_prob', 0.0):
                weight *= 2.0
            real_cost += weight

            # 2. Adversario
            if adversary_pos is not None:
                if adversary_pos == current_node or adversary_pos == next_node: caught = True
                
                # Probabilidad 50%. El adversario falla mucho mas.
                if random.random() < 0.5: 
                    try:
                        path = nx.shortest_path(env.graph, adversary_pos, next_node, weight='weight')
                        if len(path) > 1: adversary_pos = path[1]
                    except: pass
                
                if adversary_pos == next_node: caught = True

            current_node = next_node
        
        if caught:
            real_cost += 5000 
            return real_cost, False 
            
        return real_cost, True

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
            print("No se encontraron mapas reales. Usando generados.")
        return files

    def register_miners(self, miners):
        self.miners = miners
        for m in miners:
            self.stats[m.name] = {'wins': 0, 'valid': 0, 'total_cost': 0}
    
    def start_mining_round(self, block_index):
        # Generacion de semilla dinamica
        dynamic_seed = int(time.time()) + block_index * 55

        if self.real_maps:
            map_file = self.real_maps[(block_index - 1) % len(self.real_maps)]
            map_name = os.path.basename(map_file)
            env = ProblemInstance(tsplib_file=map_file, random_seed=dynamic_seed, add_traffic=True)
        else:
            map_name = f"Procedural-{block_index}"
            env = ProblemInstance(num_nodes=35, random_seed=dynamic_seed, add_traffic=True)

        threshold = NetworkValidator.calculate_dynamic_threshold(env)
        pos_adv = getattr(env, 'pos_adversario', getattr(env, 'adversary_pos', 'N/A'))

        print(f"\nBloque #{block_index} || Mapa: {map_name} ({len(env.graph.nodes)} nodos) || Threshold: {threshold:.1f}")
        print(f"Mision: {len(env.packages)} paquetes || Rival en: {pos_adv}")

        candidates = []

        for miner in self.miners:
            start = time.time()
            path = []
            
            try:
                path = miner.solve(env)
            except Exception as e:
                path = []

            duration = time.time() - start

            # Validacion
            real_cost, is_valid_logic = NetworkValidator.validate_block(env, path)
            passed_threshold = real_cost <= threshold
            is_valid_block = is_valid_logic and passed_threshold
            
            # --- CORRECCION AQUI: Usamos 'real_cost' como clave ---
            candidates.append({
                'miner': miner, 
                'path': path, 
                'real_cost': real_cost,  # <--- CAMBIADO DE 'cost' A 'real_cost'
                'valid': is_valid_block
            })

            if not is_valid_logic: status = "FAIL (Atrapado)"
            elif not passed_threshold: status = "FAIL (Ineficiente)"
            else: status = "OK"

            print(f"- {miner.name:<15} | Time: {duration:.3f}s | Real: {real_cost:.1f} {status}")
        
        # LOGICA DE GANADOR
        valid_candidates = [c for c in candidates if c['valid']]
        
        if not valid_candidates:
            # FALLBACK: Si nadie paso el threshold, gana el superviviente
            # --- CORRECCION AQUI: Usamos c['real_cost'] ---
            survivors = [c for c in candidates if c['real_cost'] < 5000 and c['path']]
            if survivors:
                print(" [AVISO] Nadie supero el umbral de eficiencia. Ganara el mejor superviviente.")
                valid_candidates = survivors

        if valid_candidates:
            # --- CORRECCION AQUI: Usamos x['real_cost'] ---
            winner = min(valid_candidates, key=lambda x: x['real_cost'])
            w_miner = winner['miner']
            
            self.chain.append(Block(block_index, datetime.now(), map_name, winner, w_miner.name, threshold))
            self.stats[w_miner.name]['wins'] += 1
            self.stats[w_miner.name]['valid'] += 1
            # --- CORRECCION AQUI: Usamos winner['real_cost'] ---
            self.stats[w_miner.name]['total_cost'] += winner['real_cost']
            
            print(f"Ganador: {w_miner.name} (Coste: {winner['real_cost']:.1f})")
        else:
            print("Bloque huerfano (Todos fueron atrapados)")

    def print_stats(self):
        print("\nRESULTADOS FINALES DE LA COMPETICION")
        print(f"{'Minero':<20} | {'Bloques':<8} | {'Coste Medio':<12}")
        for name, data in self.stats.items():
            avg = data['total_cost'] / data['valid'] if data['valid'] > 0 else 0
            print(f"{name:<20} | {data['wins']:<8} | {avg:<12.1f}")

def main():
    sim = PoUWConsensus()
    
    miners = [
        NodeAStar(),
        NodeExpectimax(),
        NodeMinimax(max_depth=3, beam_width=5),
        MinerRL()
    ]

    print("Entrenando RL (5000 episodios)...")
    train_env = ProblemInstance(num_nodes=35, random_seed=999, add_traffic=True)
    miners[3].train(train_env, episodes=5000)
    print("Entrenamiento completado.")

    sim.register_miners(miners)
    
    num_maps = len(sim.real_maps)
    rounds = max(num_maps, 5)
    
    print(f"Iniciando {rounds} rondas de mineria...")
    
    for i in range(1, rounds + 1):
        sim.start_mining_round(i)
    
    sim.print_stats()

if __name__ == "__main__":
    main()