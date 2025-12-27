import random
import numpy as np
import math
from miners import MinerAbstract
from heuristic import Heuristic

class MinerRL(MinerAbstract):
    def __init__(self, name="Supernode-RL"):
        super().__init__(name)
        self.q_table = {}  # State -> {abstract_action: q_value}
        
        # Hyperparameters
        self.gamma = 0.95
        
        # Decay parameters
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.alpha_start = 0.5
        self.alpha_end = 0.1
        
        self.trained = False
        
        # Abstract Actions
        # 0: Greedy (Minimize Distance to Target)
        # 1: Smart (Minimize Expected Cost: Dist * Traffic)
        # 2: Defensive (Maximize Distance to Adversary)
        self.actions = [0, 1, 2]

    def get_dist(self, u, v, coords):
        # Reutilizamos la distancia euclidiana de scipy que ya usa la heurística
        # o simplemente la fórmula manual. 
        # Dado que Heuristic.euclidean_distance está diseñada para (nodo, lista_paquetes),
        # aquí necesitamos distancia punto a punto.
        # Podemos usar distance.euclidean de scipy si importamos, o dejarlo manual.
        # Para consistencia y velocidad, la fórmula manual está bien, pero si quieres usar la librería:
        # return distance.euclidean(coords[u], coords[v])
        return math.sqrt((coords[u][0]-coords[v][0])**2 + (coords[u][1]-coords[v][1])**2)

    def get_closest_package(self, current_node, unvisited, coords):
        if not unvisited:
            return None
        best_p = None
        min_d = float('inf')
        for p in unvisited:
            d = self.get_dist(current_node, p, coords)
            if d < min_d:
                min_d = d
                best_p = p
        return best_p

    def get_state(self, current_node, closest_package, adversary_pos, coords, graph):
        """
        State: (Adversary_Distance_Level, Local_Traffic_Level)
        """
        # 1. Adversary Distance
        adv_dist = float('inf')
        if adversary_pos is not None:
            adv_dist = self.get_dist(current_node, adversary_pos, coords)
            
        if adv_dist < 20: # Very Close (Danger)
            adv_state = 0
        elif adv_dist < 50: # Medium (Warning)
            adv_state = 1
        else: # Far (Safe)
            adv_state = 2
            
        # 2. Local Traffic Density
        # Check average traffic probability of immediate neighbors
        neighbors = list(graph.neighbors(current_node))
        avg_traffic = 0.0
        if neighbors:
            total_traffic = sum([graph[current_node][n].get('traffic_prob', 0.0) for n in neighbors])
            avg_traffic = total_traffic / len(neighbors)
            
        if avg_traffic < 0.1: # Low Traffic
            traffic_state = 0
        elif avg_traffic < 0.3: # Medium Traffic
            traffic_state = 1
        else: # High Traffic
            traffic_state = 2
            
        return (adv_state, traffic_state)

    def map_action(self, abstract_action, current_node, closest_package, adversary_pos, graph, coords):
        neighbors = list(graph.neighbors(current_node))
        if not neighbors:
            return current_node
            
        # 0: Greedy (Distance to Target)
        if abstract_action == 0:
            if closest_package is None: return random.choice(neighbors)
            # Pick neighbor that minimizes dist to closest_package
            best_n = min(neighbors, key=lambda n: self.get_dist(n, closest_package, coords))
            return best_n
            
        # 1: Smart (Distance + Traffic)
        elif abstract_action == 1:
            if closest_package is None: return random.choice(neighbors)
            # Metric: Dist * (1 + TrafficProb*Penalty)
            def cost_metric(n):
                d = self.get_dist(n, closest_package, coords)
                traffic = graph[current_node][n].get('traffic_prob', 0.0)
                return d * (1.0 + traffic * 2.0) # Moderate penalty to guide choice
            
            best_n = min(neighbors, key=cost_metric)
            return best_n
            
        # 2: Defensive (Avoid Adversary)
        elif abstract_action == 2:
            if adversary_pos is None: return random.choice(neighbors)
            # Pick neighbor that maximizes dist to adversary
            best_n = max(neighbors, key=lambda n: self.get_dist(n, adversary_pos, coords))
            return best_n
            
        return random.choice(neighbors)

    def get_q_value(self, state, action):
        if state not in self.q_table:
            self.q_table[state] = {}
        return self.q_table[state].get(action, 0.0)

    def choose_action(self, state, epsilon):
        """
        Epsilon-greedy action selection for Abstract Actions
        """
        if random.random() < epsilon:
            return random.choice(self.actions)
        
        # Greedy
        best_q = -float('inf')
        best_actions = []
        
        for action in self.actions:
            q = self.get_q_value(state, action)
            if q > best_q:
                best_q = q
                best_actions = [action]
            elif q == best_q:
                best_actions.append(action)
        
        if not best_actions:
            return random.choice(self.actions)
        
        return random.choice(best_actions)

    def update_q_table(self, state, action, reward, next_state, alpha):
        current_q = self.get_q_value(state, action)
        
        # Max Q for next state
        max_next_q = max([self.get_q_value(next_state, a) for a in self.actions])
            
        new_q = current_q + alpha * (reward + self.gamma * max_next_q - current_q)
        
        if state not in self.q_table:
            self.q_table[state] = {}
        self.q_table[state][action] = new_q

    def train(self, problem_instance, episodes=1000):
        print(f"Entrenando {self.name} (Abstract State + Heuristic Shaping) durante {episodes} episodios...")
        
        graph = problem_instance.graph
        nodes = list(graph.nodes())
        coords = problem_instance.node_coordinates
        
        history_rewards = []

        for episode in range(episodes):
            progress = episode / episodes
            epsilon = self.epsilon_start - progress * (self.epsilon_start - self.epsilon_end)
            alpha = self.alpha_start - progress * (self.alpha_start - self.alpha_end)
            
            start_node = random.choice(nodes)
            num_packages = random.randint(3, 5)
            possible_targets = [n for n in nodes if n != start_node]
            packages = set(random.sample(possible_targets, min(num_packages, len(possible_targets))))
            adversary_pos = random.choice([n for n in nodes if n != start_node])
            
            current_node = start_node
            unvisited = packages.copy()
            
            steps = 0
            max_steps = 100
            total_reward = 0
            
            while unvisited and steps < max_steps:
                closest_package = self.get_closest_package(current_node, unvisited, coords)
                state = self.get_state(current_node, closest_package, adversary_pos, coords, graph)
                
                # Calculate Potential BEFORE move (for Shaping)
                dist_current = Heuristic.euclidean_distance(current_node, unvisited, coords)
                
                # Choose Abstract Action
                abstract_action = self.choose_action(state, epsilon)
                
                # Map to Physical Node
                next_node = self.map_action(abstract_action, current_node, closest_package, adversary_pos, graph, coords)
                
                # Execute
                edge_data = graph[current_node][next_node]
                base_cost = edge_data.get('weight', 1.0)
                traffic_prob = edge_data.get('traffic_prob', 0.0)
                
                actual_cost = base_cost
                if random.random() < traffic_prob:
                    actual_cost *= 4.0 # Heavy penalty in simulation to force learning
                
                reward = -actual_cost
                
                # Calculate Potential AFTER move (using OLD unvisited to avoid jump)
                dist_next = Heuristic.euclidean_distance(next_node, unvisited, coords)
                shaping = (dist_current - dist_next)
                reward += shaping
                
                next_unvisited = unvisited.copy()
                if next_node in next_unvisited:
                    next_unvisited.remove(next_node)
                    reward += 100
                
                if next_node == adversary_pos:
                    reward -= 500
                
                # Adversary Move
                adv_neighbors = list(graph.neighbors(adversary_pos))
                if adv_neighbors:
                    adversary_pos = random.choice(adv_neighbors)
                
                # Next State
                next_closest = self.get_closest_package(next_node, next_unvisited, coords)
                next_state = self.get_state(next_node, next_closest, adversary_pos, coords, graph)
                
                # Update Q
                self.update_q_table(state, abstract_action, reward, next_state, alpha)
                
                current_node = next_node
                unvisited = next_unvisited
                steps += 1
                total_reward += reward
            
            history_rewards.append(total_reward)
            
            log_interval = max(1, episodes // 10)
            if (episode + 1) % log_interval == 0:
                avg_reward = np.mean(history_rewards[-log_interval:])
                print(f"Episodio {episode + 1}/{episodes} | Recompensa Media: {avg_reward:.2f} | Epsilon: {epsilon:.2f}")

        self.trained = True
        print("Entrenamiento completado.")
        print("Q-Table Final (State -> Action Values):")
        for s, actions in self.q_table.items():
            best_a = max(actions, key=actions.get)
            print(f"  State {s}: Best Action {best_a} {actions}")

    def solve(self, problem_instance):
        """
        Uses the trained Q-Table to solve the specific instance.
        """
        if not self.trained:
            print("Advertencia: El agente no ha sido entrenado. Ejecutando entrenamiento rápido...")
            self.train(problem_instance, episodes=500)
            
        current_node = problem_instance.get_start_node()
        unvisited = set(problem_instance.packages)
        adversary_pos = problem_instance.pos_adversario
        graph = problem_instance.graph
        coords = problem_instance.node_coordinates
        
        path = [current_node]
        self.solution_cost = 0
        
        steps = 0
        max_steps = 200
        
        while unvisited and steps < max_steps:
            closest_package = self.get_closest_package(current_node, unvisited, coords)
            state = self.get_state(current_node, closest_package, adversary_pos, coords, graph)
            
            # Greedy action (epsilon=0)
            abstract_action = self.choose_action(state, epsilon=0)
            
            # Map
            next_node = self.map_action(abstract_action, current_node, closest_package, adversary_pos, graph, coords)
            
            # Execute
            edge_data = graph[current_node][next_node]
            base_cost = edge_data.get('weight', 1.0)
            # Note: In solve, we don't know if traffic happens until we move, 
            # but for cost calculation in 'solution_cost', we usually sum the base weights 
            # or the realized weights. Let's sum realized weights if we want to be accurate to simulation.
            # But usually 'solve' returns a plan. 
            # However, since this is an online agent (reacts to state), it's actually stepping through.
            
            # Check traffic (Simulated for the cost report)
            traffic_prob = edge_data.get('traffic_prob', 0.0)
            actual_cost = base_cost
            if random.random() < traffic_prob:
                actual_cost *= 2.0
            
            self.solution_cost += actual_cost
            
            current_node = next_node
            path.append(current_node)
            
            if current_node in unvisited:
                unvisited.remove(current_node)
                
            # Update adversary (Simulate same behavior as training or static?)
            # The problem_instance has a static pos_adversario initially. 
            # If the environment supports dynamic adversary, we should query it.
            # For now, we assume static or random as per training.
            # Let's keep it static for the 'solve' path generation unless we want to simulate the game.
            # But wait, 'solve' returns a path. If the adversary moves, the path changes.
            # The 'solve' method in other miners returns a full path list.
            # For RL, we are generating the path step-by-step.
            
            steps += 1
            
        return path
