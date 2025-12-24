import heapq
from miners import MinerAbstract
from heuristic import Heuristic
from scipy.spatial import distance

class NodeAStar(MinerAbstract):
    def __init__(self):
        super().__init__(name="Miner A*")


    def solve(self, problem_instance):
        """
        Implementación del algoritmo A* adaptado para VRP
        """    

        start_node = problem_instance.get_start_node()
        coordinates = problem_instance.get_node_coordinates()

        # convertimos la lista de paquetes en una tupla ordeanda
        # ayuda a contestar la pregunta: "qué me falta por entregar"

        package_initial = tuple(sorted(problem_instance.packages))

        # cola de prioridad
        # (f_score, g_score, current_node, path, pending_packages)
        priority_queue = []
        h_start = Heuristic.euclidean_distance(start_node, package_initial, coordinates)
        heapq.heappush(priority_queue, (h_start, 0, start_node, [start_node], package_initial))

        # hacemos un diccionario para guardar los nodos visitados
        # clave: (nodo_actual, tupla_paquetes) --> valor: menor coste g

        visited = {}

        print("Resolviendo problema...")
        while priority_queue:
            # extraemos el mejor nodo
            f, g, current_state, path, pending = heapq.heappop(priority_queue)

            # si hemos llegado al objetivo, significa que hemos encontrado la solución
            if not pending:
                self.solution_path = path
                self.solution_cost = g
                return path

            # nodos visitados
            visited_state = (current_state, pending)

            # si ya visitamos este estado con menos coste, saltamos
            if visited_state in visited and visited[visited_state] <= g:
                continue

            # guardamos el coste g    
            visited[visited_state] = g

            # expandimos el nodo
            for neighbor in problem_instance.graph.neighbors(current_state):
                # coste de viajar al vecino
                dist_to_neighbor = problem_instance.graph[current_state][neighbor]['weight']
                new_g = g + dist_to_neighbor

                # gestion del estado

                # hay que convertir la tupla a lista para poder modificar

                new_pending_list = list(pending) 

                if neighbor in new_pending_list:
                    new_pending_list.remove(neighbor)

                # convertimos la lista a tupla ordenada para sea inmutable
                new_pendint_tuple = tuple(sorted(new_pending_list))

                # calculamos la heurística
                h = Heuristic.euclidean_distance(neighbor, new_pendint_tuple, coordinates)

                # calculamos el f_score
                new_f = new_g + h 

                # añadimos el vecino a la cola de prioridad
                heapq.heappush(priority_queue, (new_f, new_g, neighbor, path + [neighbor], new_pendint_tuple))

        # si no encontramos solución, devolvemos None
        return None


                