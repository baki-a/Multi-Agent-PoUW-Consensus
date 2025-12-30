import networkx as nx
import time
from miners import MinerAbstract
from heuristic import Heuristic

class NodeMinimax(MinerAbstract):
    def __init__(self, name="NodeMinimax", time_limit=0.5): 
        super().__init__(name)
        self.time_limit = time_limit 
        self.problem = None 
        self.start_time = 0

    def _max_value(self, current, pending, adversary_pos, cost, alpha, beta, depth, coordinates):
        if time.time() - self.start_time > self.time_limit:
            raise TimeoutError()

        if depth == 0 or not pending:
            return Heuristic.evaluate_minimax(
                current, pending, adversary_pos, coordinates, cost, len(self.problem.packages)
            )

        v = float('-inf')
        neighbors = list(self.problem.graph.neighbors(current))

        for neighbor in neighbors:
            if neighbor == adversary_pos: continue

            step_cost = self.problem.graph[current][neighbor]['weight']
            new_pending = list(pending)
            if neighbor in new_pending: new_pending.remove(neighbor)
            new_pending_tuple = tuple(sorted(new_pending))

            v_new = self._min_value(neighbor, new_pending_tuple, adversary_pos, cost + step_cost, alpha, beta, depth - 1, coordinates)
            v = max(v, v_new)

            if v >= beta: return v
            alpha = max(alpha, v)

        return v

    def _min_value(self, current, pending, adversary_pos, cost, alpha, beta, depth, coordinates):
        if time.time() - self.start_time > self.time_limit:
            raise TimeoutError()

        if depth == 0 or not pending:
            return Heuristic.evaluate_minimax(
                current, pending, adversary_pos, coordinates, cost, len(self.problem.packages)
            )

        v = float('inf')
        if adversary_pos is None:
             return self._max_value(current, pending, None, cost, alpha, beta, depth - 1, coordinates)

        adv_neighbors = list(self.problem.graph.neighbors(adversary_pos))
        for adv_neighbor in adv_neighbors:
            v_new = self._max_value(current, pending, adv_neighbor, cost, alpha, beta, depth - 1, coordinates)
            v = min(v, v_new)
            if v <= alpha: return v
            beta = min(beta, v)

        return v

    def get_best_move(self, current_node, pending_packages, adversary_pos, coordinates, max_depth_limit):
        self.start_time = time.time()
        
        neighbors = list(self.problem.graph.neighbors(current_node))
        if not neighbors: return None
        
        # --- SMART FALLBACK (LA CLAVE DEL ARREGLO) ---
        # Si se acaba el tiempo, no elegimos neighbors[0] (que causa bucles).
        # Elegimos el vecino más cercano a un paquete (Greedy).
        # Esto asegura que el camión SIEMPRE AVANCE.
        best_move = min(neighbors, key=lambda n: Heuristic.euclidean_distance(n, pending_packages, coordinates))

        try:
            for d in range(1, max_depth_limit + 1):
                current_best_move = None
                current_best_val = float('-inf')
                alpha = float('-inf')
                beta = float('inf')

                for neighbor in neighbors:
                    if neighbor == adversary_pos: continue
                    
                    step_cost = self.problem.graph[current_node][neighbor]['weight']
                    temp_pending = list(pending_packages)
                    if neighbor in temp_pending: temp_pending.remove(neighbor)
                    
                    val = self._min_value(
                        neighbor, tuple(sorted(temp_pending)), adversary_pos, 
                        step_cost, alpha, beta, d - 1, coordinates
                    )

                    if val > current_best_val:
                        current_best_val = val
                        current_best_move = neighbor
                    
                    alpha = max(alpha, current_best_val)
                
                if current_best_move:
                    best_move = current_best_move
        
        except TimeoutError:
            pass # Se acabó el tiempo, usamos el best_move calculado hasta ahora (o el Greedy)
            
        return best_move

    def solve(self, problem_instance):
        print(f"Resolviendo con Minimax (Iterative Deepening, Max {self.time_limit}s/turno)...")
        
        self.problem = problem_instance
        coordinates = problem_instance.get_node_coordinates()
        start_node = problem_instance.get_start_node()
        adversary_pos = getattr(problem_instance, 'pos_adversario', getattr(problem_instance, 'adversary_pos', None))
        pending_packages = tuple(sorted(problem_instance.packages))
        
        path = [start_node]
        current_node = start_node
        total_cost = 0.0

        max_depth_cap = 4
        steps = 0 # Contador de seguridad

        while pending_packages:
            # Proteccion contra bucles infinitos
            steps += 1
            if steps > 200: 
                print(" [NodeMinimax] Límite de pasos excedido (Posible bucle). Abortando.")
                break

            best_move = self.get_best_move(current_node, pending_packages, adversary_pos, coordinates, max_depth_cap)

            if best_move is None:
                all_n = list(self.problem.graph.neighbors(current_node))
                if all_n: best_move = all_n[0]
                else: break

            total_cost += self.problem.graph[current_node][best_move]['weight']
            path.append(best_move)
            
            temp_pending = list(pending_packages)
            if best_move in temp_pending: temp_pending.remove(best_move)
            pending_packages = tuple(sorted(temp_pending))

            if adversary_pos is not None:
                try:
                    path_adv = nx.shortest_path(self.problem.graph, adversary_pos, best_move, weight='weight')
                    if len(path_adv) > 1: adversary_pos = path_adv[1]
                except: pass

            current_node = best_move

        self.solution_path = path
        self.solution_cost = total_cost
        return path