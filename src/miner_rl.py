import random
import numpy as np
import math
from miners import MinerAbstract
from heuristic import Heuristic

class MinerRL(MinerAbstract):
    def __init__(self, name="Supernode-RL"):
        super().__init__(name)
        # La Tabla Q guarda el "conocimiento" del agente.
        # Es un diccionario donde:
        #   - La clave es el ESTADO (qué está pasando).
        #   - El valor es otro diccionario con las ACCIONES y sus PUNTUACIONES (qué tan buenas son).
        self.q_table = {} 
        
        # Hiperparámetros (Configuración del aprendizaje)
        self.gamma = 0.95  # Factor de descuento: ¿Cuánto me importa el futuro? (0 = nada, 1 = mucho)
        
        # Parámetros que cambian con el tiempo (Decaimiento)
        # Epsilon: Probabilidad de explorar (hacer cosas al azar). Empieza alto y baja.
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        
        # Alpha: Tasa de aprendizaje (¿Cuánto caso le hago a la nueva información?). Empieza alto y baja.
        self.alpha_start = 0.5
        self.alpha_end = 0.1
        
        self.trained = False  # ¿Ya he entrenado?
        
        # Acciones Abstractas (Estrategias de alto nivel)
        # En lugar de elegir "ir al nodo 5", elegimos una ESTRATEGIA:
        # 0: Codicioso (Greedy) -> Ir directo al paquete más cercano.
        # 1: Inteligente (Smart) -> Ir al paquete, pero evitando atascos de tráfico.
        # 2: Defensivo -> Alejarse del ladrón (adversario).
        self.actions = [0, 1, 2]

    def get_dist(self, nodo1, nodo2, coords):
        # Calculamos la distancia en línea recta (Euclidiana) entre dos nodos.
        # Usamos el Teorema de Pitágoras: a^2 + b^2 = c^2
        x1, y1 = coords[nodo1]
        x2, y2 = coords[nodo2]
        
        distancia = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        return distancia

    def get_closest_package(self, nodo_actual, paquetes_pendientes, coords):
        # Buscamos cuál es el paquete que está más cerca de donde estamos.
        if not paquetes_pendientes:
            return None
            
        mejor_paquete = None
        distancia_minima = float('inf') # Empezamos con infinito
        
        for paquete in paquetes_pendientes:
            distancia = self.get_dist(nodo_actual, paquete, coords)
            if distancia < distancia_minima:
                distancia_minima = distancia
                mejor_paquete = paquete
                
        return mejor_paquete

    def get_state(self, nodo_actual, paquete_mas_cercano, pos_adversario, coords, graph):
        """
        Aquí definimos el ESTADO: ¿Cómo ve el mundo el agente?
        Simplificamos el mundo en dos cosas:
        1. ¿Qué tan cerca está el ladrón?
        2. ¿Hay mucho tráfico aquí al lado?
        Devolvemos una tupla: (Nivel_Peligro, Nivel_Trafico)
        """
        
        # --- 1. Distancia al Adversario (Ladrón) ---
        distancia_adversario = float('inf')
        if pos_adversario is not None:
            distancia_adversario = self.get_dist(nodo_actual, pos_adversario, coords)
            
        # Clasificamos el peligro en 3 niveles
        if distancia_adversario < 20: 
            estado_peligro = 0 # ¡Peligro! Muy cerca
        elif distancia_adversario < 50: 
            estado_peligro = 1 # Cuidado, está cerca
        else: 
            estado_peligro = 2 # Seguro, está lejos
            
        # --- 2. Densidad de Tráfico Local ---
        # Miramos a los vecinos directos para ver si hay atasco
        vecinos = list(graph.neighbors(nodo_actual))
        trafico_promedio = 0.0
        
        if vecinos:
            suma_trafico = 0
            for vecino in vecinos:
                # Obtenemos la probabilidad de tráfico de la conexión (arista)
                datos_arista = graph[nodo_actual][vecino]
                suma_trafico += datos_arista.get('traffic_prob', 0.0)
            
            trafico_promedio = suma_trafico / len(vecinos)
            
        # Clasificamos el tráfico en 3 niveles
        if trafico_promedio < 0.1: 
            estado_trafico = 0 # Tráfico Bajo (Libre)
        elif trafico_promedio < 0.3: 
            estado_trafico = 1 # Tráfico Medio
        else: 
            estado_trafico = 2 # Tráfico Alto (Atasco)
            
        return (estado_peligro, estado_trafico)

    def map_action(self, accion_abstracta, nodo_actual, paquete_mas_cercano, pos_adversario, graph, coords):
        # Esta función traduce la "Estrategia" (Acción Abstracta) en un movimiento real (ir a un nodo vecino).
        
        vecinos = list(graph.neighbors(nodo_actual))
        if not vecinos:
            return nodo_actual # Si no hay salida, nos quedamos aquí
            
        # Estrategia 0: Codicioso (Ir rápido al paquete)
        if accion_abstracta == 0:
            if paquete_mas_cercano is None: 
                return random.choice(vecinos)
            
            # Buscamos el vecino que nos deja más cerca del paquete
            mejor_vecino = None
            menor_distancia = float('inf')
            
            for vecino in vecinos:
                dist = self.get_dist(vecino, paquete_mas_cercano, coords)
                if dist < menor_distancia:
                    menor_distancia = dist
                    mejor_vecino = vecino
            return mejor_vecino
            
        # Estrategia 1: Inteligente (Balancear distancia y tráfico)
        elif accion_abstracta == 1:
            if paquete_mas_cercano is None: 
                return random.choice(vecinos)
            
            # Calculamos un "coste" para cada vecino.
            # Coste = Distancia * (1 + Penalización por Tráfico)
            mejor_vecino = None
            menor_coste = float('inf')
            
            for vecino in vecinos:
                dist = self.get_dist(vecino, paquete_mas_cercano, coords)
                prob_trafico = graph[nodo_actual][vecino].get('traffic_prob', 0.0)
                
                # Si hay mucho tráfico, el coste sube mucho
                coste = dist * (1.0 + prob_trafico * 2.0)
                
                if coste < menor_coste:
                    menor_coste = coste
                    mejor_vecino = vecino
            return mejor_vecino
            
        # Estrategia 2: Defensivo (Huir del ladrón)
        elif accion_abstracta == 2:
            if pos_adversario is None: 
                return random.choice(vecinos)
            
            # Buscamos el vecino que nos aleje MÁS del ladrón
            mejor_vecino = None
            mayor_distancia = -1.0
            
            for vecino in vecinos:
                dist = self.get_dist(vecino, pos_adversario, coords)
                if dist > mayor_distancia:
                    mayor_distancia = dist
                    mejor_vecino = vecino
            return mejor_vecino
            
        # Si pasa algo raro, elegimos al azar
        return random.choice(vecinos)

    def get_q_value(self, estado, accion):
        # Recuperamos el valor Q de la memoria. Si no existe, devolvemos 0.
        if estado not in self.q_table:
            self.q_table[estado] = {}
        return self.q_table[estado].get(accion, 0.0)

    def choose_action(self, estado, epsilon):
        """
        Elegimos qué hacer usando el método Epsilon-Greedy.
        Tiramos una moneda (random). Si sale menor que epsilon, exploramos (azar).
        Si no, explotamos (usamos lo que ya sabemos).
        """
        if random.random() < epsilon:
            return random.choice(self.actions) # Explorar: Elegir al azar
        
        # Explotar: Buscar la mejor acción en mi tabla Q
        mejor_valor_q = -float('inf')
        mejores_acciones = []
        
        for accion in self.actions:
            valor_q = self.get_q_value(estado, accion)
            
            if valor_q > mejor_valor_q:
                mejor_valor_q = valor_q
                mejores_acciones = [accion]
            elif valor_q == mejor_valor_q:
                mejores_acciones.append(accion) # Si hay empate, las guardamos todas
        
        # Si no sabemos nada, elegimos al azar
        if not mejores_acciones:
            return random.choice(self.actions)
        
        # Si hay varias igual de buenas, elegimos una de ellas al azar
        return random.choice(mejores_acciones)

    def update_q_table(self, estado, accion, recompensa, proximo_estado, alpha):
        # Esta es la fórmula mágica de Q-Learning.
        # Actualizamos el valor de la acción que acabamos de tomar.
        
        valor_actual_q = self.get_q_value(estado, accion)
        
        # Miramos cuál es la mejor acción posible desde el siguiente estado (para estimar el futuro)
        max_q_futuro = -float('inf')
        for a in self.actions:
            q = self.get_q_value(proximo_estado, a)
            if q > max_q_futuro:
                max_q_futuro = q
                
        # Fórmula: NuevoQ = ViejoQ + TasaAprendizaje * (Sorpresa)
        # Sorpresa = (Recompensa + LoQueEsperoDelFuturo - LoQueCreiaAntes)
        nuevo_valor_q = valor_actual_q + alpha * (recompensa + self.gamma * max_q_futuro - valor_actual_q)
        
        # Guardamos el nuevo valor
        if estado not in self.q_table:
            self.q_table[estado] = {}
        self.q_table[estado][accion] = nuevo_valor_q

    def train(self, problem_instance, episodes=1000):
        print(f"Entrenando {self.name} durante {episodes} episodios...")
        
        graph = problem_instance.graph
        nodes = list(graph.nodes())
        coords = problem_instance.node_coordinates
        
        historial_recompensas = []

        for episodio in range(episodes):
            # Calculamos epsilon y alpha dinámicos (van bajando con el tiempo)
            progreso = episodio / episodes
            epsilon = self.epsilon_start - progreso * (self.epsilon_start - self.epsilon_end)
            alpha = self.alpha_start - progreso * (self.alpha_start - self.alpha_end)
            
            # Preparamos un escenario aleatorio para entrenar
            nodo_inicio = random.choice(nodes)
            num_paquetes = random.randint(3, 5)
            posibles_destinos = [n for n in nodes if n != nodo_inicio]
            paquetes = set(random.sample(posibles_destinos, min(num_paquetes, len(posibles_destinos))))
            pos_adversario = random.choice([n for n in nodes if n != nodo_inicio])
            
            nodo_actual = nodo_inicio
            paquetes_pendientes = paquetes.copy()
            
            pasos = 0
            max_pasos = 100
            recompensa_total = 0
            
            while paquetes_pendientes and pasos < max_pasos:
                # 1. Observar el estado
                paquete_cercano = self.get_closest_package(nodo_actual, paquetes_pendientes, coords)
                estado = self.get_state(nodo_actual, paquete_cercano, pos_adversario, coords, graph)
                
                # Calculamos el "Potencial" ANTES de movernos (para ayudar al aprendizaje)
                distancia_total_antes = Heuristic.euclidean_distance(nodo_actual, paquetes_pendientes, coords)
                
                # 2. Elegir acción
                accion_abstracta = self.choose_action(estado, epsilon)
                
                # 3. Ejecutar acción (Moverse)
                proximo_nodo = self.map_action(accion_abstracta, nodo_actual, paquete_cercano, pos_adversario, graph, coords)
                
                # Calcular coste del movimiento
                datos_arista = graph[nodo_actual][proximo_nodo]
                coste_base = datos_arista.get('weight', 1.0)
                prob_trafico = datos_arista.get('traffic_prob', 0.0)
                
                coste_real = coste_base
                # Simulamos si nos toca tráfico (mala suerte)
                if random.random() < prob_trafico:
                    coste_real *= 4.0 # Penalización fuerte para que aprenda a evitarlo
                
                # La recompensa es negativa (es un coste)
                recompensa = -coste_real
                
                # Recompensa por acercarse al objetivo (Shaping)
                distancia_total_despues = Heuristic.euclidean_distance(proximo_nodo, paquetes_pendientes, coords)
                mejora = (distancia_total_antes - distancia_total_despues)
                recompensa += mejora
                
                # Gestionar paquetes entregados
                nuevos_paquetes_pendientes = paquetes_pendientes.copy()
                if proximo_nodo in nuevos_paquetes_pendientes:
                    nuevos_paquetes_pendientes.remove(proximo_nodo)
                    recompensa += 100 # ¡Premio por entregar!
                
                # Penalización si nos pilla el ladrón
                if proximo_nodo == pos_adversario:
                    recompensa -= 500
                
                # Mover al adversario (aleatorio)
                vecinos_adv = list(graph.neighbors(pos_adversario))
                if vecinos_adv:
                    pos_adversario = random.choice(vecinos_adv)
                
                # 4. Observar el nuevo estado
                proximo_paquete_cercano = self.get_closest_package(proximo_nodo, nuevos_paquetes_pendientes, coords)
                proximo_estado = self.get_state(proximo_nodo, proximo_paquete_cercano, pos_adversario, coords, graph)
                
                # 5. Aprender (Actualizar tabla Q)
                self.update_q_table(estado, accion_abstracta, recompensa, proximo_estado, alpha)
                
                # Avanzar
                nodo_actual = proximo_nodo
                paquetes_pendientes = nuevos_paquetes_pendientes
                pasos += 1
                recompensa_total += recompensa
            
            historial_recompensas.append(recompensa_total)
            
            # Mostrar progreso cada cierto tiempo
            intervalo_log = max(1, episodes // 10)
            if (episodio + 1) % intervalo_log == 0:
                recompensa_media = np.mean(historial_recompensas[-intervalo_log:])
                print(f"Episodio {episodio + 1}/{episodes} | Recompensa Media: {recompensa_media:.2f} | Epsilon: {epsilon:.2f}")

        self.trained = True
        print("Entrenamiento completado.")
        print("Tabla Q Final (Estado -> Valores de Acción):")
        for s, actions in self.q_table.items():
            best_a = max(actions, key=actions.get)
            print(f"  Estado {s}: Mejor Acción {best_a} {actions}")

    def solve(self, problem_instance):
        """
        Usa la Tabla Q ya entrenada para resolver el problema real.
        """
        if not self.trained:
            print("Advertencia: El agente no ha sido entrenado. Entrenando rápido...")
            self.train(problem_instance, episodes=500)
            
        nodo_actual = problem_instance.get_start_node()
        paquetes_pendientes = set(problem_instance.packages)
        pos_adversario = problem_instance.pos_adversario
        graph = problem_instance.graph
        coords = problem_instance.node_coordinates
        
        ruta = [nodo_actual]
        self.solution_cost = 0
        
        pasos = 0
        max_pasos = 200
        
        while paquetes_pendientes and pasos < max_pasos:
            paquete_cercano = self.get_closest_package(nodo_actual, paquetes_pendientes, coords)
            estado = self.get_state(nodo_actual, paquete_cercano, pos_adversario, coords, graph)
            
            # Elegimos la mejor acción (Greedy, epsilon=0) porque ya no estamos aprendiendo
            accion_abstracta = self.choose_action(estado, epsilon=0)
            
            # Traducimos la estrategia a un movimiento real
            proximo_nodo = self.map_action(accion_abstracta, nodo_actual, paquete_cercano, pos_adversario, graph, coords)
            
            # Calculamos costes (Simulados para el reporte)
            datos_arista = graph[nodo_actual][proximo_nodo]
            coste_base = datos_arista.get('weight', 1.0)
            prob_trafico = datos_arista.get('traffic_prob', 0.0)
            
            coste_real = coste_base
            if random.random() < prob_trafico:
                coste_real *= 2.0
            
            self.solution_cost += coste_real
            
            nodo_actual = proximo_nodo
            ruta.append(nodo_actual)
            
            if nodo_actual in paquetes_pendientes:
                paquetes_pendientes.remove(nodo_actual)
                
            pasos += 1
            
        return ruta
