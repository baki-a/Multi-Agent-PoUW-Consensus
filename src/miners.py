from abc import ABC, abstractmethod

class MinerAbstract(ABC):
    def __init__(self, name="Miner"):
        self.name = name
        self.solution_path = []
        self.solution_cost = 0

    @abstractmethod
    def solve(self, problem_instance):
        """
        Recibe una instancia del problema: mapa y misión
        Debe devolver una lista de nodos que representa la solución 
        """
        pass
