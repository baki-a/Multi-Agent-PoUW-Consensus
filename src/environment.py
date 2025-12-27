import networkx as nx
import numpy as np 
import random 
from scipy.spatial import distance

class ProblemInstance:

    def __init__(self, tsplib_file=None, num_nodes=20, random_seed=None, add_traffic=True):
        """
        Inicializa una instancia de un problema
        Si se da un archivo real, carga datos reales. Si no, genera aleatorios para testing rapido
        """

        if random_seed:
            random.seed(random_seed)
            np.random.seed(random_seed)

        self.graph = nx.Graph()
        self.adversary_position = None
        self.adversary_target = None
        self.node_coordinates = {}

        # 1. Capa Base. NP-hard --> carga el mapa físico
        loaded = False
        if tsplib_file:
            loaded = self.load_from_tsplib(tsplib_file)

        if not loaded and not tsplib_file:
            self.generate_random(num_nodes)
        elif not loaded and tsplib_file:
            print(f"Error: no se pudo cargar el archivo {tsplib_file}")
            return

        # 2. Capa estocástica. Con incertidumbre. Añade tráfico
        if add_traffic:
            self.add_traffic_layer()

        # 3. Capa de adversario. Añade al rival
        self.add_adversary_layer()

        self.packages = self.generate_mission(num_packages = 5)

    def generate_mission(self, num_packages):
        """
        Genera una misión con num_packages paquetes al azar
        """
        possible_nodes = list(self.graph.nodes())
        start_node = self.get_start_node()
        if start_node in possible_nodes:
            possible_nodes.remove(start_node)

        count = min(num_packages, len(possible_nodes))
        return random.sample(possible_nodes, count)
        

    def load_from_tsplib(self, filename):
        """
        Lee un archivo estándar .tsp y construye el grafo a base de sus datos
        Extrae las coordenades 
        """
        lectura_coordenadas = False

        try:
            with open(filename, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"Error: El archivo {filename} no se encuentra.")
            return False
        
        for line in lines:
            line = line.strip()

            # detectamos el inicio de sección de coordenadas
            if "NODE_COORD_SECTION" in line:
                lectura_coordenadas = True
                continue

            # detectamos el final de sección de coordenadas
            if "EOF" in line:
                lectura_coordenadas = False
                break

            # leemos las coordenadas
            if lectura_coordenadas:
                partes = line.split()
                # saltamos lineas vacias
                if len(partes) < 3:
                    continue

                try:
                    # partes[0] es el id del nodo
                    # partes[1] es la coordenada x
                    # partes[2] es la coordenada y***
                    node_id = int(partes[0])
                    x = float(partes[1])
                    y = float(partes[2])
                    self.node_coordinates[node_id] = (x, y)
                except ValueError:
                    print(f"Error: línea inválida: {line}")
                    continue
        if len(self.node_coordinates) == 0:
            print(f"Error: no se encontraron coordenadas en el archivo {filename}")
            return False
        
        # Añadimos nodos y aristas al grafo
        nodes = list(self.node_coordinates.keys())
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                u, v = nodes[i], nodes[j]

                # Distancia euclidiana como coste base
                dist = distance.euclidean(self.node_coordinates[u], self.node_coordinates[v])
                self.graph.add_edge(u, v, weight=dist, traffic_prob=0.0)

        print(f"Mapa cargado desde {filename}: {len(nodes)} ciudades")
        return True


    def generate_random(self, num_nodes):
        """
        Genera un grafo aleatorio con num_nodes nodos
        """
        self.node_coordinates = {i: (np.random.rand() * 100, np.random.rand() * 100) for i in range(num_nodes)}
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                dist = distance.euclidean(self.node_coordinates[i], self.node_coordinates[j])
                self.graph.add_edge(i, j, weight=dist, traffic_prob=0.0)
        
        print(f"Mapa generado aleatoriamente: {num_nodes} ciudades")

    def add_traffic_layer(self):
        """
        Capa 2. Selecciona aleatoriamente aristas y les asigna probabilidad de tráfico.
        traffic_prob = probabilidad de que el coste se duplique
        """

        edges = list(self.graph.edges(data=True))
        num_traffic_zones = int(len(edges) * 0.2)  # quiere decir que el 20% de las aristas tendrán tráfico

        chosen_edges = random.sample(edges, num_traffic_zones) 
        for u, v, data in chosen_edges:
            # Añadimos la probabilidad de tráfico
            probabilidad = round(random.uniform(0.1, 0.5), 2)
            self.graph[u][v]['traffic_prob'] = probabilidad    


    def add_adversary_layer(self):
        """
        Capa 3. Coloca el adversario en un nodo aleatorio
        """

        nodes = list(self.graph.nodes())
        self.pos_adversario = random.choice(nodes)
        # el objetivo del adversario es llegar al nodo objetivo del agente
        self.obj_adversario = random.choice(nodes)

    def get_start_node(self):
        return list(self.graph.nodes())[0]  # el agente empieza en el primer nodo

    def print_graph(self):
        print(self.graph)

    def get_node_coordinates(self):
        return self.node_coordinates

# Pruebas rapidas
if __name__ == "__main__":
    # random 
    env_random = ProblemInstance(num_nodes=20, random_seed=42)
    env_random.print_graph()

    print("\n Inspección de grafo")
    print(f"Nodos: {env_random.graph.number_of_nodes()}")
    print(f"Aristas: {env_random.graph.number_of_edges()}")
    print(f"Adversario: {env_random.pos_adversario}")

    # Inspeccionar una arista con tráfico
    edges_with_traffic = [(u, v, d) for u, v, d in env_random.graph.edges(data=True) if d['traffic_prob'] > 0]
    print(f"Aristas (carreteras) con tráfico: {len(edges_with_traffic)}")

    u, v, data = edges_with_traffic[0]
    print(f"Arista: {u} -> {v} | Coste: {data['weight']} | Tráfico: {data['traffic_prob']}")

    # file berlin52.tsp
    env_berlin = ProblemInstance(tsplib_file="data/berlin52.tsp")
    env_berlin.print_graph()

    print("\n Inspección de grafo berlin52")
    print(f"Nodos: {env_berlin.graph.number_of_nodes()}")
    print(f"Aristas: {env_berlin.graph.number_of_edges()}")
    print(f"Adversario: {env_berlin.pos_adversario}")

    # Inspeccionar una arista con tráfico
    edges_with_traffic = [(u, v, d) for u, v, d in env_berlin.graph.edges(data=True) if d['traffic_prob'] > 0]
    print(f"Aristas (carreteras) con tráfico: {len(edges_with_traffic)}")

    u, v, data = edges_with_traffic[0]
    print(f"Arista: {u} -> {v} | Coste: {data['weight']} | Tráfico: {data['traffic_prob']}")

    

                    