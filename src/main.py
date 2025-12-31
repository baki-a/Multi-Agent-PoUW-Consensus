import time
import random
import os
import glob
import sys
from datetime import datetime
import networkx as nx
from environment import ProblemInstance
from heuristic import Heuristic
from miner_astar import NodeAStar
from miner_expectimax import NodeExpectimax
from miner_minimax import NodeMinimax
from miner_rl import MinerRL


COST_SIMILARITY_THRESHOLD = 0.05  # umbral en caso de de empate técnico


MAX_NODES_FOR_DEMO = 130  # Filtramos mapas gigantes para que la presentación sea fluida

class Block:
    def __init__(self, block_idx, created_at, problem_name, solution_data, miner_name, threshold):
        self.index = block_idx
        self.timestamp = created_at
        self.problem_id = problem_name
        self.solution_path = solution_data['path']
        self.real_cost = solution_data['real_cost']
        self.threshold = threshold
        self.miner_name = miner_name
        self.hash = hash(f"{block_idx}{created_at}{miner_name}{solution_data['real_cost']}")

class NetworkValidator:
    @staticmethod
    def calculate_dynamic_threshold(env):
        """Calcula el umbral de dificultad basado en una heurística Greedy x4."""
    
        start_position = env.get_start_node()
        coordinates = env.node_coordinates

        # si usamos env.packages directamente, el problema es que 
        # no es un alista, por lo tanto, no se puede modificar
        pending_packages = list(env.packages)

        distance_estimated = 0.0
        current_position = start_position

        # el bucle principal del metodo
        while pending_packages:
            
            # encontramos al vecino mas cercano
            shortest_distance = float('inf')
            next_node = None

            # recorremos todos los vecinos para ver cual es el mas cercano
            for package in pending_packages:
                # para una aprox usamos la distancia euclideana
                distance = Heuristic.euclidean_distance(current_position, [package], coordinates) # pasamos en formato lista poruque espera una lista

                if distance < shortest_distance:
                    shortest_distance = distance
                    next_node = package

                # actualizamos el estado y sumamos la distancia de este vecino 
                distance_estimated += shortest_distance

                # nos movemos a ese vecino
                current_position = next_node

                # eliminamos el vecino de la lista de vecinos pendientes
                pending_packages.remove(next_node)
        
        # tenemos en cuenta para el cálculo del umbral que
        # multiplicamos por 4.0, ya que la distancia euclideana es perfecta 
        # asi pues, el agente real necesitará más pasos para esquivar obstaculos 
        threshold = distance_estimated * 4.0

        if threshold < 3000.0:
            threshold = 3000.0

        return threshold
                

    @staticmethod
    def validate_block(env, route):
        """
        Simula la ruta en el entorno real (Tráfico + Adversario).
        Devuelve (coste_real, sobrevivio_sin_ser_atrapado).
        """
        if not route: return float('inf'), False

        real_cost = 0.0
        current_node = route[0]
        # Obtener adversario de forma segura
        adversary_pos = getattr(env, 'pos_adversario', getattr(env, 'adversary_pos', None))
        got_caught = False
        coords = env.node_coordinates

        for i in range(1, len(route)):
            next_node = route[i]
            
            # 1. Capa Estocástica (Tráfico)
            edge = env.graph[current_node][next_node]
            weight = edge['weight']
            if random.random() < edge.get('traffic_prob', 0.0):
                weight *= 2.0
            real_cost += weight

            # 2. Capa Adversaria (Rival)
            if adversary_pos is not None:
                if adversary_pos == current_node or adversary_pos == next_node:
                    got_caught = True
                
                # --- LÓGICA DE PERSECUCIÓN AJUSTADA ---
                # Pasamos [adversary_pos] como lista para evitar errores en heuristic.py
                dist_to_rival = Heuristic.euclidean_distance(current_node, [adversary_pos], coords)
                
                # Radio de visión: 700. Probabilidad de acierto: 60%.
                # Esto es suficiente para castigar a A* si pasa muy cerca,
                # pero permite a Minimax escapar si mantiene la distancia.
                is_near = (dist_to_rival < 900)
                moves_successfully = (random.random() < 0.85) 

                if is_near and moves_successfully:
                    try:
                        path = nx.shortest_path(env.graph, adversary_pos, next_node, weight='weight')
                        if len(path) > 1: adversary_pos = path[1]
                    except: pass
                else:
                    # Si no nos ve, patrulla aleatoriamente
                    if random.random() < 0.5:
                        neighbors = list(env.graph.neighbors(adversary_pos))
                        if neighbors: adversary_pos = random.choice(neighbors)
                
                if adversary_pos == next_node:
                    got_caught = True

            current_node = next_node
        
        # Penalización masiva por muerte
        if got_caught:
            real_cost += 5000
            return real_cost, False
            
        return real_cost, True

class PoUWConsensus:
    def __init__(self):
        self.chain = []
        self.miners = []
        self.stats = {}
        self.round_history = []
        self.real_maps = self._load_real_maps()
        
    def _load_real_maps(self):
        """Carga mapas filtrando los problemáticos y los demasiado grandes."""
        all_files = glob.glob("data/*.tsp") + glob.glob("*.tsp")
        valid_files = []
        
        print("\n" + "="*60)
        print("  ESCANEO DE MAPAS DISPONIBLES")
        print("="*60)
        
        for f in all_files:
            try:
                # Pre-lectura para verificar tamaño y formato
                with open(f, 'r') as file:
                    content = file.read()
                    # Rechazar mapas sin coordenadas explícitas (causa del error bayg29)
                    if "NODE_COORD_SECTION" not in content and "DISPLAY_DATA_SECTION" not in content:
                         print(f"  [X] {os.path.basename(f):<20} -> Formato no soportado (sin coords)")
                         continue

                    # Verificar tamaño
                    for line in content.split('\n'):
                        if "DIMENSION" in line:
                            parts = line.split(':')
                            if len(parts) >= 2:
                                node_count = int(parts[1].strip())
                                if node_count < MAX_NODES_FOR_DEMO + 10:
                                    print(f"  [OK] {os.path.basename(f):<20} ({node_count} nodos)")
                                    valid_files.append(f)
                                else:
                                    print(f"  [--] {os.path.basename(f):<20} ({node_count} nodos) -> Demasiado grande")
                            break
            except Exception as e:
                print(f"  [!!] Error leyendo {f}: {e}")

        if not valid_files:
            print("  [!!] No se encontraron mapas válidos. Usando generados.")
        
        print("="*60 + "\n")
        return valid_files

    def register_miners(self, miners):
        self.miners = miners
        for m in miners:
            self.stats[m.name] = {'wins': 0, 'clean_wins': 0, 'total_cost': 0.0}
    
    def _pick_winner(self, candidates, is_rescue):
        """
        LÓGICA DE DESEMPATE MEJORADA:
        1. Identificamos el coste mínimo global.
        2. Seleccionamos TODOS los candidatos que estén dentro del umbral de similitud
           (COST_SIMILARITY_THRESHOLD) respecto a ese mínimo.
        3. De ese grupo de "finalistas", gana el que tenga el MENOR tiempo.
        """
        if not candidates: return None
        
        # 1. Encontrar el mejor coste base (el "Gold Standard" de esta ronda)
        min_cost = min(c['real_cost'] for c in candidates)
        
        # Evitar división por cero
        baseline = max(min_cost, 1.0)
        
        # 2. Filtrar "Finalistas": Cualquiera cuyo coste no se aleje más del X% del mejor
        #    Si el coste es idéntico, la diferencia es 0%, así que entra seguro.
        finalists = []
        for c in candidates:
            diff_percent = (c['real_cost'] - min_cost) / baseline
            if diff_percent <= COST_SIMILARITY_THRESHOLD:
                finalists.append(c)
        
        # 3. Desempate: De los finalistas, gana el más rápido (menor solve_time)
        #    Ordenamos por tiempo ascendente y cogemos el primero.
        winner = min(finalists, key=lambda x: x['solve_time'])
        
        # Logging informativo si hubo "robo" por velocidad
        # (Si el ganador por tiempo no es el que tenía el coste mínimo absoluto)
        if winner['real_cost'] > min_cost:
             print(f"  ⚡ DESEMPATE: {winner['miner'].name} gana por VELOCIDAD "
                   f"({winner['solve_time']:.6f}s) a pesar de mayor coste "
                   f"({winner['real_cost']:.1f} vs {min_cost:.1f})")
        elif len(finalists) > 1 and winner['real_cost'] == min_cost:
             # Caso de empate exacto en coste
             print(f"  ⚡ EMPATE EXACTO: {winner['miner'].name} gana por ser el más rápido ({winner['solve_time']:.6f}s)")

        return winner

    def start_mining_round(self, block_idx):
        dynamic_seed = int(time.time()) + block_idx * 55
        
        # Selección de mapa
        map_name = "Procedural"
        if self.real_maps:
            map_file = self.real_maps[(block_idx - 1) % len(self.real_maps)]
            map_name = os.path.basename(map_file)
            
        try:
            # Intentamos cargar entorno (Protección contra fallos de carga)
            if self.real_maps:
                env = ProblemInstance(tsplib_file=map_file, random_seed=dynamic_seed, add_traffic=True)
            else:
                map_name = f"Procedural-{block_idx}"
                env = ProblemInstance(num_nodes=35, random_seed=dynamic_seed, add_traffic=True)

            # Check extra por si el grafo está vacío
            if not env.graph or len(env.graph.nodes) == 0:
                raise ValueError("Grafo vacío o error de carga.")

            threshold = NetworkValidator.calculate_dynamic_threshold(env)
            adv_node = getattr(env, 'pos_adversario', getattr(env, 'adversary_pos', 'N/A'))

            print(f"\nBLOQUE #{block_idx} | {map_name} ({len(env.graph.nodes)} nodos) | Threshold: {threshold:.0f}")
            print(f"Misión: {len(env.packages)} paquetes | Rival en: {adv_node}")
            print("-" * 75)
            print(f"{'MINERO':<18} | {'TIEMPO':>8} | {'COSTE REAL':>10} | ESTADO")
            print("-" * 75)

            candidates = []
            for miner in self.miners:
                start = time.perf_counter()
                path = []
                try:
                    path = miner.solve(env)
                except: path = []
                duration = time.perf_counter() - start
                
                real_cost, survived = NetworkValidator.validate_block(env, path)
                passed = real_cost <= threshold
                valid = survived and passed
                
                candidates.append({
                    'miner': miner, 'path': path, 'real_cost': real_cost,
                    'solve_time': duration, 'valid': valid, 'caught': not survived
                })
                
                if not survived: status = "FAIL (Atrapado)"
                elif not passed: status = "FAIL (Ineficiente)"
                else: status = "OK"
                
                print(f"{miner.name:<18} | {duration:>8.6f}s | {real_cost:>10.1f} | {status}")

            print("-" * 75)

            # Lógica de Selección
            valid_candidates = [c for c in candidates if c['valid']]
            is_rescue = False
            
            # Protocolo de Emergencia
            if not valid_candidates:
                is_rescue = True
                survivors = [c for c in candidates if c['path']]
                # Preferimos los que sobrevivieron aunque fueran ineficientes
                clean_survivors = [c for c in survivors if not c['caught']]
                
                if clean_survivors:
                    valid_candidates = clean_survivors
                elif survivors:
                    print("  [⚠️] Todos atrapados. Se elegirá el mejor intento disponible.")
                    valid_candidates = survivors

            if valid_candidates:
                # Usamos la nueva función de desempate
                winner = self._pick_winner(valid_candidates, is_rescue)
                w_miner = winner['miner']
                
                win_type = "VICTORIA LIMPIA" if not is_rescue else "RESCATE"
                
                self.chain.append(Block(block_idx, datetime.now(), map_name, winner, w_miner.name, threshold))
                self.stats[w_miner.name]['wins'] += 1
                if not is_rescue: self.stats[w_miner.name]['clean_wins'] += 1
                self.stats[w_miner.name]['total_cost'] += winner['real_cost']
                
                self.round_history.append({
                    'round': block_idx, 'winner': w_miner.name, 
                    'type': win_type, 'cost': winner['real_cost']
                })
                
                print(f"  🏆 GANADOR: {w_miner.name} (Coste: {winner['real_cost']:.1f}) [{win_type}]")
            else:
                print("  [💀] BLOQUE HUÉRFANO FINAL (Error crítico)")

        except Exception as e:
            print(f"  [⚠️] SALTANDO BLOQUE #{block_idx} ({map_name}): {e}")

    def print_stats(self):
        print("\n" + "="*60)
        print("  CLASIFICACIÓN FINAL PO-UW")
        print("="*60)
        print(f"  {'MINERO':<20} | {'BLOQUES':>8} | {'LIMPIAS':>8} | {'COSTE MEDIO':>12}")
        print("-" * 60)
        for name, data in sorted(self.stats.items(), key=lambda x: x[1]['wins'], reverse=True):
            avg = data['total_cost'] / max(data['wins'], 1)
            print(f"  {name:<20} | {data['wins']:>8} | {data['clean_wins']:>8} | {avg:>12.1f}")
        print("="*60)

def main():
    sim = PoUWConsensus()
    
    # Configuración de mineros
    miners = [
        NodeAStar(),
        NodeExpectimax(),
        # Minimax con 0.5s para darle un poco más de margen a pensar
        NodeMinimax(time_limit=0.5), 
        MinerRL()
    ]

    print("\n  >>> ENTRENANDO SUPERNODE-RL (3000 EPISODIOS) <<<")
    train_env = ProblemInstance(num_nodes=35, random_seed=999, add_traffic=True)
    miners[3].train(train_env, episodes=3000)
    print("  >>> ENTRENAMIENTO COMPLETADO <<<\n")

    sim.register_miners(miners)
    
    rounds = max(len(sim.real_maps), 5)
    
    for i in range(1, rounds + 1):
        sim.start_mining_round(i)
    
    sim.print_stats()

if __name__ == "__main__":
    main()