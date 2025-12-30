import networkx as nx
import heapq
from miners import MinerAbstract
from heuristic import Heuristic

class NodeMinimax(MinerAbstract):
    def __init__(self, name="NodeMinimax", max_depth=3, beam_width=5):
        super().__init__(name)
        self.max_depth = max_depth
        self.beam_width = beam_width # Solo miramos los X mejores vecinos
        self.problem = None 

    def _get_promising_neighbors(self, current_node, pending_packages, coordinates):
        """
        Optimización Clave: En lugar de devolver los 200 vecinos de un mapa grande,
        devuelve solo los 'beam_width' (ej: 5) más prometedores.
        Criterio: Los que nos acercan más a los paquetes (menor distancia heurística).
        """
        all_neighbors = list(self.problem.graph.neighbors(current_node))
        
        # Si hay pocos vecinos, devolvemos todos y no perdemos tiempo ordenando
        if len(all_neighbors) <= self.beam_width:
            return all_neighbors

        # Ordenamos los vecinos: preferimos los que tienen menor distancia a los paquetes
        # Esto es una pre-evaluación rápida para podar el árbol
        scored_neighbors = []
        for n in all_neighbors:
            # Distancia heurística simple al paquete más cercano
            h = Heuristic.euclidean_distance(n, pending_packages, coordinates)
            # Coste del paso (peso de la arista)
            g = self.problem.graph[current_node][n]['weight']
            score = h + g
            scored_neighbors.append((score, n))
        
        # Ordenar de menor a mayor coste y quedarse con los mejores
        scored_neighbors.sort(key=lambda x: x[0])
        best_neighbors = [n for score, n in scored_neighbors[:self.beam_width]]
        
        return best_neighbors

    def _max_value(self, current, pending, adversary_pos, cost, alpha, beta, depth, coordinates):
        """ Turno MAX (Nosotros) """
        if depth == 0 or not pending:
            return Heuristic.evaluate_minimax(
                current, pending, adversary_pos, coordinates, cost, len(self.problem.packages)
            )

        v = float('-inf')
        
        # USAMOS LA OPTIMIZACIÓN AQUÍ
        neighbors = self._get_promising_neighbors(current, pending, coordinates)

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
        """ Turno MIN (Adversario) """
        if depth == 0 or not pending:
            return Heuristic.evaluate_minimax(
                current, pending, adversary_pos, coordinates, cost, len(self.problem.packages)
            )

        v = float('inf')

        if adversary_pos is None:
             return self._max_value(current, pending, None, cost, alpha, beta, depth - 1, coordinates)

        # Para el adversario también limitamos, o se haría eterno
        # El adversario intentará ir hacia nosotros (Minimizar distancia a current)
        # Nota: Aquí simplificamos y usamos los vecinos directos del grafo para no complicar la heurística inversa
        # pero limitamos la cantidad si son demasiados.
        adv_neighbors = list(self.problem.graph.neighbors(adversary_pos))
        if len(adv_neighbors) > self.beam_width:
            adv_neighbors = adv_neighbors[:self.beam_width]

        for adv_neighbor in adv_neighbors:
            v_new = self._max_value(current, pending, adv_neighbor, cost, alpha, beta, depth - 1, coordinates)
            v = min(v, v_new)

            if v <= alpha: return v
            beta = min(beta, v)

        return v

    def solve(self, problem_instance):
        # Ajuste dinámico: Si el mapa es gigante, reducimos profundidad para asegurar respuesta
        current_depth = self.max_depth
        num_nodes = len(problem_instance.graph.nodes)
        if num_nodes > 30:
            current_depth = 2 # Bajamos profundidad en mapas enormes
            
        print(f"Resolviendo con Minimax (Profundidad {current_depth}, Beam {self.beam_width})...")
        
        self.problem = problem_instance
        coordinates = problem_instance.get_node_coordinates()
        start_node = problem_instance.get_start_node()
        adversary_pos = getattr(problem_instance, 'pos_adversario', getattr(problem_instance, 'adversary_pos', None))
        pending_packages = tuple(sorted(problem_instance.packages))
        
        path = [start_node]
        current_node = start_node
        total_cost = 0.0

        while pending_packages:
            
            best_move = None
            best_val = float('-inf')
            alpha = float('-inf')
            beta = float('inf')

            # Aplicamos la optimización también en la raíz
            neighbors = self._get_promising_neighbors(current_node, pending_packages, coordinates)

            for neighbor in neighbors:
                if neighbor == adversary_pos: continue

                step_cost = self.problem.graph[current_node][neighbor]['weight']
                temp_pending = list(pending_packages)
                if neighbor in temp_pending: temp_pending.remove(neighbor)
                
                val = self._min_value(
                    neighbor, tuple(sorted(temp_pending)), adversary_pos, 
                    total_cost + step_cost, alpha, beta, current_depth - 1, coordinates
                )
                
                if val > best_val:
                    best_val = val
                    best_move = neighbor
                
                alpha = max(alpha, best_val)

            if best_move is None:
                # Si la poda nos dejó sin movimientos (raro), cogemos cualquiera válido
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
                    ruta_rival = nx.shortest_path(self.problem.graph, adversary_pos, best_move, weight='weight')
                    if len(ruta_rival) > 1: adversary_pos = ruta_rival[1]
                except: pass

            current_node = best_move

        self.solution_path = path
        self.solution_cost = total_cost
        return path