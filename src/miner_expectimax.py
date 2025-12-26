import heapq
from miners import MinerAbstract
from heuristic import Heuristic

class NodeExpectimax(MinerAbstract):
    def __init__(self, name="NodeExpectimax"):
        super().__init__(name)

    def expected_cost(self, u, v, graph):
        """
        Calcula la media ponderada de los costes de las aristas incidentes en v
        La formula es: coste = base + (base * probablidad de trafico * penalización))
        """
        peso_base = graph[u][v]['weight']
        probabilidad_trafico = graph[u][v].get('traffic_prob', 0.0)
        
        # por ahora, la penalización es de factor 1
        # un atasco añade 100% extra de tiempo
        penalizacion = 1.0

        coste_esperado = peso_base + (peso_base * probabilidad_trafico * penalizacion)
        return coste_esperado
        

    def solve(self, problem_instance):
        """
        Resuelve el problema utilizando el algoritmo expectimax
        """
        start_node = problem_instance.get_start_node()
        coordinates = problem_instance.get_node_coordinates()
        packages = tuple(sorted(problem_instance.packages)) 

        cola_prioridad = []

        # usamos heuristica de distancia euclidea 
        heuristica_start = Heuristic.euclidean_distance(start_node, packages, coordinates)
        heapq.heappush(cola_prioridad, (heuristica_start, 0.0, start_node, [start_node], packages))
        dict_visitados = {}

        print("Calculando ruta considerando tráfico")

        while cola_prioridad:
            f_value, g_value, current_state, path, pending_packages = heapq.heappop(cola_prioridad)

            # caso meta
            if not pending_packages:
                self.solution_path = path
                self.solution_cost = g_value
                return path

            # caso visitados
            estado_visitado = (current_state, pending_packages)
            if estado_visitado in dict_visitados:
                continue
            dict_visitados[estado_visitado] = g_value

            # vecinos
            for neighbor in problem_instance.graph[current_state]:
                step_cost = self.expected_cost(current_state, neighbor, problem_instance.graph)
                new_g_value = g_value + step_cost

                # gestionamos estado
                new_pending_packages = list(pending_packages)
                if neighbor in new_pending_packages:
                    new_pending_packages.remove(neighbor)
                new_pending_packages_tuple = tuple(sorted(new_pending_packages))

                heuristica_vecino = Heuristic.euclidean_distance(neighbor, new_pending_packages_tuple, coordinates)
                new_f_value = new_g_value + heuristica_vecino

                heapq.heappush(cola_prioridad, (new_f_value, new_g_value, neighbor, path + [neighbor], new_pending_packages_tuple))

        return None