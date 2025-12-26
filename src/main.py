from environment import ProblemInstance
from miner_astar import NodeAStar
from miner_expectimax import NodeExpectimax
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

def run_experiment(name, env):
    print(f"Iniciando experimento {name}")

    # ejecutamos a*
    miner_astar = NodeAStar()

    start_time = time.time()
    path_astar = miner_astar.solve(env)
    end_time = time.time()

    time_astar = end_time - start_time
    
    # ejecutamos expectimax
    miner_expectimax = NodeExpectimax()

    start_time = time.time()
    path_expectimax = miner_expectimax.solve(env)
    end_time = time.time()

    time_expectimax = end_time - start_time

    # imprimimos resultados
    print(f"Resultados experimento {name}")
    print(f"A*: {path_astar} \nCoste: {miner_astar.solution_cost:.2f}\nTiempo: {time_astar:.2f}")
    print(f"Expectimax: {path_expectimax} \nCoste: {miner_expectimax.solution_cost:.2f}\nTiempo: {time_expectimax:.2f}")
    
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
    else:
        print(f"Cargando el mapa real {tsp_file}")
        env_no_traffic = ProblemInstance(tsplib_file=tsp_file, random_seed=42, add_traffic=False)
    run_experiment("No traffic", env_no_traffic)

    print("Cargando el mapa con tráfico")
    if not os.path.exists(tsp_file):
        print(f"Error: el archivo {tsp_file} no se encuentra")
        env_traffic = ProblemInstance(num_nodes=20, random_seed=42, add_traffic=True)
    else:
        print(f"Cargando el mapa real {tsp_file}")
        env_traffic = ProblemInstance(tsplib_file=tsp_file, random_seed=42, add_traffic=True)
    env_traffic.packages = env_no_traffic.packages
    run_experiment("Traffic", env_traffic)
    
    


if __name__ == "__main__":
    #test_system()
    main_experiment()
