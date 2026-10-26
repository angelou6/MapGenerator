#Aqui empieza el jale de oswaldo
"""
Oswaldo hizo esto, oswaldo intentaba hacerlo bien mientras su mente pensaba en comida... no deberias confiar en 
oswaldo mientras piensa en comida
"""
# Oswaldo importo las funciones que ya habian hecho para usarlas con los nuevos mapas y luego importo el random
from generator.map_utils import (
    generate_random_map,
    calculate_distribution_penalty,
    calculate_enemy_penalty,
    calculate_path_penalty,
    calculate_reward_penalty,
    get_difficulty
)
import random

# Oswaldo importo la funcion de busqueda de camino para poder calcular la dificultad del mapa
from path_search.breadth import bfs

class Genetic:
    def __init__(self, population_size=100):
        """
        Oswaldo, el genio de la genética de mapas, ha decidido crear una clase para manejar la evolución de mapas.
        :param population_size: es el tamaño de la poblacion de mapas de cada generacion,
        oswaldo piensa que 100 es un buen numero
        """
        self.population_size = population_size
        self.population = []  # oswaldo guardo la lista de mapas aqui, oswaldo dice que esa es la pobracion inicial
        self.generation = 1   # este es un contador de generaciones, oswealdo piensa que suena estupido, pero quien es
        #el para para decir lo que es estupido y lo que no

    def generate_initial_population(self):
        """
        Oswaldo ha decidido crear la poblacion inicial aqui, son completamente aleatorios
        """
        self.population = []
        
        for _ in range(self.population_size):
            # 1. Oswaldo decidio generar un mapa aleatorio usando la que ya tenemos
            nuevo_mapa = generate_random_map()
            
            # 2. Oswaldo empaqueta el mapa en un diccionario, este mapa sera un individuo
            individuo = {
                "matrix": nuevo_mapa,
                "fitness": None  # Oswaldo ya sabe cómo calcularlo, ocurre en la función de abajo
            }
            
            # 3. Ahora oswaldo agrega el individuo a la poblacion
            self.population.append(individuo)
            
        print(f"Población inicial generada: {len(self.population)} mapas listos.")
        return self.population

    def evaluar_poblacion(self):
        """
        Oswaldo agarro el pedo luego de comer e hizo esto para evaluar que tan buenos son los mapas 
        Fitness = Dificultad base - Penalizaciones.
        """
        for individuo in self.population:
            matriz_del_mapa = individuo["matrix"]
            
            # 1. Oswaldo manda a buscar el camino real con su confiable BFS (en realidad no confio en el, se porta
            # raro a veces, hace rato lo vio comiendo un cheto con salsa verde, quien hace eso?)
            resultado_bfs = bfs(matriz_del_mapa)
            
            # 2. Oswaldo decidio calcular todas las penalizaciones aqui, pasandole el resultado de la busqueda
            penalidad_distribucion = calculate_distribution_penalty(matriz_del_mapa)
            penalidad_enemigos = calculate_enemy_penalty(matriz_del_mapa)
            penalidad_caminos = calculate_path_penalty(matriz_del_mapa, solution=resultado_bfs)
            penalidad_recompensas = calculate_reward_penalty(matriz_del_mapa)
            
            suma_total_penalizaciones = (
                penalidad_distribucion + 
                penalidad_enemigos + 
                penalidad_caminos + 
                penalidad_recompensas
            )
            
            # 3. Oswaldo calcula la dificultad del mapa en bruto usando la ruta de verdad
            dificultad_base = (
                get_difficulty(matriz_del_mapa, solution=resultado_bfs)
                if resultado_bfs is not None
                else 0
            )
            
            # 4. Oswaldo calcula el fitness final (Oswaldo multiplica por 10 para que si el mapa es muy malo, 
            # lo mande pero a chingar a su madre)
            calificacion_final = dificultad_base - (suma_total_penalizaciones * 10)
            
            # 5. Oswaldo guarda la calificacion final en el individuo, y les recuerda que el individuo es un mapa de la 
            # poblacion, por eso es un individuo, jeje
            individuo["fitness"] = calificacion_final

    def torneo_seleccion(self, tamaño_torneo=3):
        """
        Oswaldo se embriago de poder, y ahora hizo un club de la pelea para los mapas.
        elige el tamaño de torno y participantes al azar y sobrevive el mejor

        primera regla del club de la pelea: siempre habla del club de la pelea

        """
        # Oswaldo selecciona los mapas participantes al azar para el torneo, aunque puedes entrar 
        # mas facilmente si tienes una hamburguesa con su nombre
        participantes = random.sample(self.population, tamaño_torneo)
        
        # Oswaldo no decide quien gana, el mejor mapa es el que gana, lo siento bobos, aqui no hay 
        #anvorguesas que los puedan ayudar
        padre_ganador = max(participantes, key=lambda individuo: individuo["fitness"])
        
        return padre_ganador

# --- Zona de pruebas  ---
if __name__ == "__main__":
    ag = Genetic(population_size=10)
    poblacion = ag.generate_initial_population()
    
    # Oswaldo manda a evaluar la poblacion recien hecha
    ag.evaluar_poblacion()
    
    print(f"El primer individuo ahora tiene un fitness de: {poblacion[0]['fitness']}")
    
    # E-e-e-es hora de-de-de-del DUELO
    padre_1 = ag.torneo_seleccion()
    padre_2 = ag.torneo_seleccion()
    
    print(f"El Padre 1 ganador del torneo tiene un fitness de: {padre_1['fitness']}")
    print(f"El Padre 2 ganador del torneo tiene un fitness de: {padre_2['fitness']}")

#Aqui termina el trabajo de oswaldo, por ahora, oswaldo consiguio comida mientras terminaba esto, pero no hubo coca