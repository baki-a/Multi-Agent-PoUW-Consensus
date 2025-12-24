from scipy.spatial import distance

class Heuristic:
    @staticmethod
    def euclidean_distance(current_node, pending_packages, coordinates):
        """
        Calcula la distancia al pquete pendiente más cercanao
        """

        if not pending_packages:
            return 0.0

        current_position = coordinates[current_node]

        # Calculamso la distancia a todos los paquetes que faltan y nos quedamos 
        # con la más cercana

        # En otras palabras: en los mejores de los casos, iré directo al más cercano

        minimum_distance = float('inf')

        for package in pending_packages:
            package_pos = coordinates[package]
            dist = distance.euclidean(current_position, package_pos)
            if dist < minimum_distance:
                minimum_distance = dist
        return minimum_distance
