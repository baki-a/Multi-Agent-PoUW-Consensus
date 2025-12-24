from environment import ProblemInstance
from miner_astar import NodeAStar
import os

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


if __name__ == "__main__":
    test_system()
