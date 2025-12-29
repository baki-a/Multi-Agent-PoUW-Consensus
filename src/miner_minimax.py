import networkx as nx
from miners import MinerAbstract
from heuristic import Heuristic 

class NodeMinimax(MinerAbstract):
    def __init__(self, name="NodeMinimax", max_depth=3):
        super().__init__(name)
        self.max_depth = max_depth
        self.problem = None # Guardamos referencia al problema para acceder al grafo

    def _max_value(self, current, pending, adversary_pos, cost, alpha, beta, depth, coordinates):
        """ Turno MAX (nuestro agente) """
        # Caso base: Llamamos a la heurística externa
        if depth == 0 or not pending:
            return Heuristic.evaluate_minimax(
                current, pending, adversary_pos, coordinates, cost, len(self.problem.packages)
            )

        v = float('-inf')
        
        for neighbor in self.problem.graph.neighbors(current):
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
        """ Turno MIN (Adversario) """
        if depth == 0 or not pending:
            return Heuristic.evaluate_minimax(
                current, pending, adversary_pos, coordinates, cost, len(self.problem.packages)
            )

        v = float('inf')

        if adversary_pos is None:
             return self._max_value(current, pending, None, cost, alpha, beta, depth - 1, coordinates)

        for adv_neighbor in self.problem.graph.neighbors(adversary_pos):
            v_new = self._max_value(current, pending, adv_neighbor, cost, alpha, beta, depth - 1, coordinates)
            v = min(v, v_new)

            if v <= alpha: return v
            beta = min(beta, v)

        return v

    def solve(self, problem_instance):
        print(f"Resolviendo con Minimax (Profundidad {self.max_depth})...")
        
        self.problem = problem_instance  # Guardamos referencia
        coordinates = problem_instance.get_node_coordinates()
        start_node = problem_instance.get_start_node()
        adversary_pos = problem_instance.pos_adversario
        pending_packages = tuple(sorted(problem_instance.packages))
        
        path = [start_node]
        current_node = start_node
        total_cost = 0.0

        while pending_packages:
            
            # decision minimax
            best_move = None
            best_val = float('-inf')
            alpha = float('-inf')
            beta = float('inf')

            for neighbor in self.problem.graph.neighbors(current_node):
                if neighbor == adversary_pos: continue

                step_cost = self.problem.graph[current_node][neighbor]['weight']
                temp_pending = list(pending_packages)
                if neighbor in temp_pending: temp_pending.remove(neighbor)
                
                # Llamada recursiva
                val = self._min_value(
                    neighbor, tuple(sorted(temp_pending)), adversary_pos, 
                    total_cost + step_cost, alpha, beta, self.max_depth - 1, coordinates
                )
                
                if val > best_val:
                    best_val = val
                    best_move = neighbor
                
                alpha = max(alpha, best_val)

            if best_move is None:
                break

            # ejecución
            total_cost += self.problem.graph[current_node][best_move]['weight']
            path.append(best_move)
            
            temp_pending = list(pending_packages)
            if best_move in temp_pending: temp_pending.remove(best_move)
            pending_packages = tuple(sorted(temp_pending))

            # simulación rival
            if adversary_pos is not None:
                try:
                    ruta_rival = nx.shortest_path(self.problem.graph, adversary_pos, best_move, weight='weight')
                    if len(ruta_rival) > 1: adversary_pos = ruta_rival[1]
                except: pass

            current_node = best_move

        self.solution_path = path
        self.solution_cost = total_cost
        return path