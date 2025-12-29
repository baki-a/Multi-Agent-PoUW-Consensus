from environment import ProblemInstance
from miner_astar import NodeAStar
from miner_expectimax import NodeExpectimax
from miner_rl import MinerRL
import os
import time

def test_system():

    print("Iniciando test del sistema")

    # cargamos el mapa real
    tsp_file = "data/berlin52.tsp"

    if not os.path.exists(tsp_file):
        print(f"Error: el archivo {tsp_file} no se encuentra")
        env = ProblemInstance(num_nodes=20, random_seed=42)
    else:
        print(f"Cargando el mapa real {tsp_file}")
        env = ProblemInstance(tsplib_file=tsp_file, random_seed=42)

    # imprimimos la misión para verificar
    start_node = env.get_start_node()
    print(f"Inicio en nodo: {start_node}")
    print(f"Paquetes a entregar ({len(env.packages)}): {env.packages}")

    # inicializamos el miner
    miner = NodeAStar()

    # solucionamos
    path = miner.solve(env)

    if path:
        print(f"SOLUCIÓN ENCONTRADA")
        print(f"Ruta: {path}")
        print(f"Coste: {miner.solution_cost:.2f}")

        # verficación manual
        missed = [p for p in env.packages if p not in path]
        if missed:
            print(f"Se han perdido los paquetes: {missed}")
        else:
            print("Se han entregado todos los paquetes")
    else:
        print("No se encontró solución")

def calculate_real_cost(env, path):
    """
    Simula el recorrido de una ruta y calcula el coste real sufriendo el tráfico estocástico.
    """
    if not path: return float('inf')
    
    total_cost = 0.0
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        edge_data = env.graph[u][v]
        base_cost = edge_data.get('weight', 1.0)
        traffic_prob = edge_data.get('traffic_prob', 0.0)
        
        # Simulación de tráfico
        import random
        actual_cost = base_cost
        if random.random() < traffic_prob:
            actual_cost *= 2.0
            
        total_cost += actual_cost
    return total_cost

def run_experiment(name, env, miner_rl=None):
    print(f"Iniciando experimento {name}")

    # ejecutamos a*
    miner_astar = NodeAStar()

    start_time = time.time()
    path_astar = miner_astar.solve(env)
    end_time = time.time()
    time_astar = end_time - start_time
    
    # Coste Real A*
    real_cost_astar = calculate_real_cost(env, path_astar)
    
    # ejecutamos expectimax
    miner_expectimax = NodeExpectimax()

    start_time = time.time()
    path_expectimax = miner_expectimax.solve(env)
    end_time = time.time()
    time_expectimax = end_time - start_time
    
    # Coste Real Expectimax
    real_cost_expectimax = calculate_real_cost(env, path_expectimax)

    # imprimimos resultados
    print(f"Resultados experimento {name}")
    print(f"A*: {path_astar}")
    print(f"  Coste Planificado: {miner_astar.solution_cost:.2f}")
    print(f"  Coste REAL (Simulado): {real_cost_astar:.2f}")
    print(f"  Tiempo: {time_astar:.2f}")
    
    print(f"Expectimax: {path_expectimax}")
    print(f"  Coste Planificado: {miner_expectimax.solution_cost:.2f}")
    print(f"  Coste REAL (Simulado): {real_cost_expectimax:.2f}")
    print(f"  Tiempo: {time_expectimax:.2f}")
    
    if miner_rl:
        start_time = time.time()
        path_rl = miner_rl.solve(env)
        end_time = time.time()
        time_rl = end_time - start_time
        # El coste de RL ya se calcula simulado dentro de su solve, pero para ser justos recalculamos igual
        real_cost_rl = calculate_real_cost(env, path_rl)
        
        print(f"RL: {path_rl}")
        print(f"  Coste REAL (Simulado): {real_cost_rl:.2f}")
        print(f"  Tiempo: {time_rl:.2f}")

    if time_expectimax > time_astar:
        diff = time_expectimax - time_astar
        print(f"El algoritmo expectimax tarda {diff:.2f} segundos más que A*")
    else:
        diff = time_astar - time_expectimax
        print(f"El algoritmo A* tarda {diff:.2f} segundos más que expectimax")


    if path_astar and path_expectimax:
        print("Se han entregado todos los paquetes")
    else:
        print("No se han entregado todos los paquetes")

def main_experiment():
    print("Pruebas experimentales")
    
    # cargamos el mapa real
    tsp_file = "data/berlin52.tsp"
    
    print("Cargando el mapa real")
    if not os.path.exists(tsp_file):
        print(f"Error: el archivo {tsp_file} no se encuentra")
        env_no_traffic = ProblemInstance(num_nodes=20, random_seed=42, add_traffic=False)
        env_traffic = ProblemInstance(num_nodes=20, random_seed=42, add_traffic=True)
    else:
        print(f"Cargando el mapa real {tsp_file}")
        env_no_traffic = ProblemInstance(tsplib_file=tsp_file, random_seed=42, add_traffic=False)
        env_traffic = ProblemInstance(tsplib_file=tsp_file, random_seed=42, add_traffic=True)
    
    # Entrenamos el agente RL una vez con el entorno más complejo (con tráfico)
    print("Entrenando Agente RL (Supernode)...")
    miner_rl = MinerRL()
    miner_rl.train(env_traffic, episodes=10000)

    run_experiment("No traffic", env_no_traffic, miner_rl)

    print("Cargando el mapa con tráfico")
    env_traffic.packages = env_no_traffic.packages
    run_experiment("Traffic", env_traffic, miner_rl)
    
    


if __name__ == "__main__":
    #test_system()
    main_experiment()

