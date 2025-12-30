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
            
        return max(estimated_dist * 4.0, 3000.0)

    @staticmethod
    def validate_block(env, route):
        if not route: return float('inf'), False

        real_cost = 0.0
        current_node = route[0]
        adversary_pos = getattr(env, 'pos_adversario', getattr(env, 'adversary_pos', None))
        caught = False
        coords = env.node_coordinates

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
                
                # --- ARREGLO TYPE ERROR ---
                # Pasamos [adversary_pos] (lista) en vez de un entero suelto
                dist_to_rival = Heuristic.euclidean_distance(current_node, [adversary_pos], coords)
                
                # Persecución inteligente limitada
                should_chase = (dist_to_rival < 600) and (random.random() < 0.5)

                if should_chase:
                    try:
                        path = nx.shortest_path(env.graph, adversary_pos, next_node, weight='weight')
                        if len(path) > 1: adversary_pos = path[1]
                    except: pass
                else:
                    if random.random() < 0.5:
                        neighbors = list(env.graph.neighbors(adversary_pos))
                        if neighbors: adversary_pos = random.choice(neighbors)
                
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
        # Filtro de mapas para evitar esperas eternas en mapas gigantes
        all_files = glob.glob("data/*.tsp") + glob.glob("*.tsp")
        valid_files = []
        
        print(f"🔎 Analizando {len(all_files)} mapas candidatos...")
        for f in all_files:
            try:
                with open(f, 'r') as file:
                    for line in file:
                        if "DIMENSION" in line:
                            parts = line.split(':')
                            if len(parts) >= 2:
                                dim = int(parts[1].strip())
                                # Solo aceptamos mapas < 150 nodos para esta demo
                                if dim < 150: 
                                    valid_files.append(f)
                                    print(f"  ✅ {os.path.basename(f)} aceptado ({dim} nodos)")
                                else:
                                    print(f"  ❌ {os.path.basename(f)} descartado (Demasiado grande)")
                            break
            except: pass

        if not valid_files:
            print("⚠️ No se encontraron mapas pequeños. Usando generados.")
        return valid_files

    def register_miners(self, miners):
        self.miners = miners
        for m in miners:
            self.stats[m.name] = {'wins': 0, 'valid': 0, 'total_cost': 0}
    
    def start_mining_round(self, block_index):
        dynamic_seed = int(time.time()) + block_index * 55

        # Selección del mapa
        map_name = "Procedural"
        if self.real_maps:
            map_file = self.real_maps[(block_index - 1) % len(self.real_maps)]
            map_name = os.path.basename(map_file)
            
        try:
            # Intentamos cargar el entorno
            if self.real_maps:
                env = ProblemInstance(tsplib_file=map_file, random_seed=dynamic_seed, add_traffic=True)
            else:
                map_name = f"Procedural-{block_index}"
                env = ProblemInstance(num_nodes=35, random_seed=dynamic_seed, add_traffic=True)

            # --- CORRECCIÓN DE SEGURIDAD ---
            # Si el grafo está vacío (falló la carga), lanzamos error para saltar esta ronda
            if not env.graph or len(env.graph.nodes) == 0:
                raise ValueError(f"El mapa {map_name} no se cargó correctamente (Grafo vacío).")

            threshold = NetworkValidator.calculate_dynamic_threshold(env)
            pos_adv = getattr(env, 'pos_adversario', getattr(env, 'adversary_pos', 'N/A'))

            print(f"\nBloque #{block_index} || Mapa: {map_name} ({len(env.graph.nodes)} nodos) || Threshold: {threshold:.1f}")
            print(f"Mision: {len(env.packages)} paquetes || Rival en: {pos_adv}")

            # ... (Resto de la lógica de mineros: candidates, bucle for, validación...)
            # Copia aquí el resto del código de start_mining_round que tenías
            # desde "candidates = []" hasta el final del método.
            
            candidates = []
            for miner in self.miners:
                # ... (tu código de ejecución de mineros) ...
                start = time.time()
                path = []
                try:
                    path = miner.solve(env)
                except Exception as e:
                    path = []
                duration = time.time() - start
                
                real_cost, is_valid_logic = NetworkValidator.validate_block(env, path)
                passed_threshold = real_cost <= threshold
                is_valid_block = is_valid_logic and passed_threshold
                
                candidates.append({
                    'miner': miner, 
                    'path': path, 
                    'real_cost': real_cost, 
                    'valid': is_valid_block,
                    'caught': not is_valid_logic
                })
                
                if not is_valid_logic: status = "FAIL (Atrapado)"
                elif not passed_threshold: status = "FAIL (Ineficiente)"
                else: status = "OK"
                print(f"- {miner.name:<15} | Time: {duration:.3f}s | Real: {real_cost:.1f} {status}")

            # Selección del ganador
            valid_candidates = [c for c in candidates if c['valid']]
            forced_win = False
            if not valid_candidates:
                survivors = [c for c in candidates if c['path']]
                clean_survivors = [c for c in survivors if not c['caught']]
                if clean_survivors:
                    valid_candidates = clean_survivors
                elif survivors:
                    print(" [AVISO] Todos atrapados. Ganará el mártir más eficiente.")
                    valid_candidates = survivors
                    forced_win = True

            if valid_candidates:
                winner = min(valid_candidates, key=lambda x: x['real_cost'])
                w_miner = winner['miner']
                note = " (RESCATADO)" if forced_win else ""
                self.chain.append(Block(block_index, datetime.now(), map_name, winner, w_miner.name, threshold))
                self.stats[w_miner.name]['wins'] += 1
                if not forced_win: self.stats[w_miner.name]['valid'] += 1
                self.stats[w_miner.name]['total_cost'] += winner['real_cost']
                print(f"Ganador: {w_miner.name} (Coste: {winner['real_cost']:.1f}){note}")
            else:
                print("Bloque huérfano FINAL")

        except Exception as e:
            print(f"⚠️ SALTANDO BLOQUE #{block_index} ({map_name}): {e}")
            # El programa continuará con el siguiente bloque en lugar de detenerse

    def print_stats(self):
        print("\nRESULTADOS FINALES DE LA COMPETICIÓN")
        print(f"{'Minero':<20} | {'Bloques':<8} | {'Coste Medio':<12}")
        for name, data in self.stats.items():
            div = data['valid'] if data['valid'] > 0 else (data['wins'] if data['wins'] > 0 else 1)
            avg = data['total_cost'] / div
            print(f"{name:<20} | {data['wins']:<8} | {avg:<12.1f}")

def main():
    sim = PoUWConsensus()
    
    miners = [
        NodeAStar(),
        NodeExpectimax(),
        NodeMinimax(time_limit=0.3), # Limite de tiempo para evitar lags
        MinerRL()
    ]

    print("Entrenando RL (3000 episodios)...")
    train_env = ProblemInstance(num_nodes=35, random_seed=999, add_traffic=True)
    miners[3].train(train_env, episodes=3000)
    print("Entrenamiento completado.")

    sim.register_miners(miners)
    
    # Ejecutamos rondas segun mapas disponibles
    num_maps = len(sim.real_maps)
    rounds = max(num_maps, 5)
    
    print(f"Iniciando {rounds} rondas de minería...")
    for i in range(1, rounds + 1):
        sim.start_mining_round(i)
    
    sim.print_stats()

if __name__ == "__main__":
    main()