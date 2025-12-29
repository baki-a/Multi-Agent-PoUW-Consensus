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

    @staticmethod
    def heuristic_minimax(current_node, pending_packages, adversary_pos, coordinates, total_cost, total_packages_count):
        """
        Función de utilidad que devuleve un número para decir si va ganando o perdiendo dependiendo de 4 factores:
        - Cantidad de paquetes pendientes
        - Distancia al paquete más cercano
        - Coste total
        - Posición del adversario
        """

        # paquetes entregados, es el objetivo 
        packages_delivered = total_packages_count - len(pending_packages)
        score = packages_delivered * 1000

        # penalizamos teniendo en cuenta la distancia al paquete más cercano
        distance_to_closest_package = Heuristic.euclidean_distance(current_node, pending_packages, coordinates)  # usamos la distancia euclideana 
        # para penalizar más cuando el paquete está lejos
        # hay penalización porque queremos que la distancia sea 0
        score -= distance_to_closest_package # nos aseguramos de que, en caso de que no se pueda entregar, que se acerque lo más posible

        if adversary_pos is not None:
            # calculamos la distancia al enemigo
            current_position_agent = coordinates[current_node]
            current_position_adversary = coordinates[adversary_pos]
            distance_to_adversary = distance.euclidean(current_position_agent, current_position_adversary)

            # penalizamos teniendo lo tanto que está cerca de nosotros
            if distance_to_adversary < 2.0:
                score -= 5000  # estamos muy pero que muy cerca del enemigo
            else:
                score -= distance_to_adversary * 10

        # penalizamos teniendo en cuenta el coste total
        score -= total_cost * 10

        return score


