import networkx as nx
import numpy as np 
import random 
import copy
from scipy.spatial import distance

class ProblemInstance:

    def __init__(self, tsplib_file=None, num_nodes=20, random_seed=None, k_neighbors=None):
        """
        Inicializa una instancia de un problema
        Si se da un archivo real, carga datos reales. Si no, genera aleatorios para testing rapido
        
        Args:
            tsplib_file: Path to TSPLIB format file
            num_nodes: Number of nodes for random generation
            random_seed: Seed for reproducibility
            k_neighbors: If set, creates K-nearest-neighbor sparse graph. None = fully connected
        """

        if random_seed:
            random.seed(random_seed)
            np.random.seed(random_seed)

        self.graph = nx.Graph()
        self.adversary_position = None
        self.adversary_target = None
        self.node_coordinates = {}
        self.k_neighbors = k_neighbors

        # 1. Capa Base. NP-hard --> carga el mapa físico
        loaded = False
        if tsplib_file:
            loaded = self.load_from_tsplib(tsplib_file)

        if not loaded and not tsplib_file:
            self.generate_random(num_nodes)
        elif not loaded and tsplib_file:
            print(f"Error: no se pudo cargar el archivo {tsplib_file}")
            return

        # Apply sparse graph if k_neighbors specified
        if self.k_neighbors:
            self._apply_knn_sparsity()

        # 2. Capa estocástica. Con incertidumbre. Añade tráfico
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

    # ==================== K-NEAREST-NEIGHBOR SPARSE GRAPH ====================
    
    def _apply_knn_sparsity(self):
        """
        Convert fully connected graph to K-nearest-neighbor sparse graph.
        Keeps only the k nearest neighbors for each node, creating choke points.
        """
        if not self.k_neighbors:
            return
            
        k = self.k_neighbors
        nodes = list(self.graph.nodes())
        edges_to_keep = set()
        
        for node in nodes:
            # Get distances to all other nodes
            distances = []
            for other in nodes:
                if other != node and self.graph.has_edge(node, other):
                    dist = self.graph[node][other]['weight']
                    distances.append((dist, other))
            
            # Sort by distance and keep k nearest
            distances.sort()
            for dist, neighbor in distances[:k]:
                # Add both directions to set (undirected graph)
                edge = tuple(sorted([node, neighbor]))
                edges_to_keep.add(edge)
        
        # Remove edges not in the keep set
        edges_to_remove = []
        for u, v in self.graph.edges():
            edge = tuple(sorted([u, v]))
            if edge not in edges_to_keep:
                edges_to_remove.append((u, v))
        
        for u, v in edges_to_remove:
            self.graph.remove_edge(u, v)
        
        # Ensure graph is connected - add minimum spanning tree edges if needed
        if not nx.is_connected(self.graph):
            # Get all original distances and add MST edges
            components = list(nx.connected_components(self.graph))
            while len(components) > 1:
                # Find closest nodes between components
                min_dist = float('inf')
                best_edge = None
                for c1 in components[0]:
                    for c2 in components[1]:
                        dist = distance.euclidean(
                            self.node_coordinates[c1], 
                            self.node_coordinates[c2]
                        )
                        if dist < min_dist:
                            min_dist = dist
                            best_edge = (c1, c2)
                
                if best_edge:
                    self.graph.add_edge(best_edge[0], best_edge[1], 
                                       weight=min_dist, traffic_prob=0.0)
                components = list(nx.connected_components(self.graph))
        
        print(f"Grafo convertido a K-NN sparse (k={k}): {self.graph.number_of_edges()} aristas")

    # ==================== STOCHASTIC LAYER METHODS ====================
    
    def get_stochastic_cost(self, u, v, sample=True):
        """
        Get the travel cost between two nodes considering traffic probability.
        
        Args:
            u, v: Edge endpoints
            sample: If True, randomly sample based on traffic_prob.
                    If False, return expected cost.
        
        Returns:
            float: Actual or expected travel cost
        """
        if not self.graph.has_edge(u, v):
            return float('inf')
        
        edge_data = self.graph[u][v]
        base_cost = edge_data['weight']
        traffic_prob = edge_data.get('traffic_prob', 0.0)
        
        if sample:
            # Stochastic sampling: traffic doubles the cost
            if random.random() < traffic_prob:
                return base_cost * 2.0
            return base_cost
        else:
            # Expected value: E[cost] = (1-p)*base + p*(2*base) = base*(1+p)
            return base_cost * (1.0 + traffic_prob)
    
    def get_expected_cost(self, u, v):
        """Get expected cost considering traffic probability."""
        return self.get_stochastic_cost(u, v, sample=False)
    
    def get_edges_with_traffic(self):
        """Return list of edges that have traffic probability > 0."""
        return [(u, v, d) for u, v, d in self.graph.edges(data=True) 
                if d.get('traffic_prob', 0) > 0]

    # ==================== ADVERSARIAL LAYER METHODS ====================
    
    def get_adversary_position(self):
        """Get current adversary position."""
        return self.pos_adversario
    
    def set_adversary_position(self, node):
        """Set adversary position (for simulation)."""
        if node in self.graph.nodes():
            self.pos_adversario = node
    
    def get_adversary_neighbors(self):
        """Get all possible moves for the adversary."""
        return list(self.graph.neighbors(self.pos_adversario))
    
    def move_adversary_toward(self, target_node):
        """
        Move adversary one step toward target using shortest path.
        Returns the new position.
        """
        if self.pos_adversario == target_node:
            return self.pos_adversario
        
        try:
            path = nx.shortest_path(self.graph, self.pos_adversario, target_node, weight='weight')
            if len(path) > 1:
                self.pos_adversario = path[1]
        except nx.NetworkXNoPath:
            pass  # No path exists, adversary stays in place
        
        return self.pos_adversario
    
    def move_adversary_greedy(self, miner_position, miner_destination):
        """
        Adversary moves greedily to intercept miner.
        Tries to get between miner's current position and destination.
        """
        neighbors = self.get_adversary_neighbors()
        if not neighbors:
            return self.pos_adversario
        
        best_neighbor = None
        best_score = float('inf')
        
        for neighbor in neighbors:
            # Score based on shortest path distances
            try:
                dist_to_dest = nx.shortest_path_length(self.graph, neighbor, miner_destination, weight='weight')
            except nx.NetworkXNoPath:
                dist_to_dest = float('inf')
            
            try:
                dist_to_miner = nx.shortest_path_length(self.graph, neighbor, miner_position, weight='weight')
            except nx.NetworkXNoPath:
                dist_to_miner = float('inf')
            
            # Prefer positions close to both miner and destination
            score = min(dist_to_dest, dist_to_miner)
            if score < best_score:
                best_score = score
                best_neighbor = neighbor
        
        if best_neighbor:
            self.pos_adversario = best_neighbor
        
        return self.pos_adversario
    
    def is_blocked_by_adversary(self, node):
        """Check if a node is blocked by the adversary."""
        return node == self.pos_adversario
    
    def is_edge_blocked(self, u, v):
        """Check if traversing edge u->v would be blocked by adversary at v."""
        return self.pos_adversario == v
    
    def get_safe_neighbors(self, node):
        """Get neighbors that are not blocked by adversary."""
        return [n for n in self.graph.neighbors(node) 
                if not self.is_blocked_by_adversary(n)]

    # ==================== GAME STATE METHODS ====================
    
    def clone(self):
        """Create a deep copy of the environment for simulation."""
        new_env = ProblemInstance.__new__(ProblemInstance)
        new_env.graph = self.graph.copy()
        new_env.node_coordinates = self.node_coordinates.copy()
        new_env.pos_adversario = self.pos_adversario
        new_env.obj_adversario = self.obj_adversario
        new_env.packages = self.packages.copy()
        new_env.adversary_position = self.adversary_position
        new_env.adversary_target = self.adversary_target
        new_env.k_neighbors = self.k_neighbors
        return new_env
    
    def get_state(self, current_node, pending_packages):
        """
        Get a hashable state representation for search algorithms.
        """
        return (current_node, tuple(sorted(pending_packages)), self.pos_adversario)
    
    def apply_action(self, current_node, next_node, pending_packages):
        """
        Apply miner action and return new state.
        Returns: (new_pending, cost, blocked)
        """
        new_pending = list(pending_packages)
        if next_node in new_pending:
            new_pending.remove(next_node)
        
        blocked = self.is_blocked_by_adversary(next_node)
        cost = self.graph[current_node][next_node]['weight'] if not blocked else float('inf')
        
        return new_pending, cost, blocked

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

    

                    