from collections import deque
from typing import Annotated

import numpy as np
from numpy._typing import NDArray

from path_search.utils import get_ajacent_blockers
from representation import BLOCKERS, MAP_SHAPE, Elemento

#oswaldo estuvo aqui
import random
import os
import json
#oswaldo se fue de aqui

rng = np.random.default_rng()

CONSTRAINTS: Annotated[
    dict[Elemento, tuple[int, int]], "Rango de objectos posibles dentro del mapa."
] = {
    Elemento.OBSTACULO: (20, 35),
    Elemento.ENEMIGO: (5, 10),
    Elemento.RECOMPENZA: (3, 6),
}

FANCY_ELEMENTOS = {
    Elemento.TRANSITABLE: ".",
    Elemento.OBSTACULO: "#",
    Elemento.ENEMIGO: "E",
    Elemento.RECOMPENZA: "R",
    Elemento.INICIO: "I",
    Elemento.META: "M",
}


def generate_random_map() -> NDArray[np.int32]:
    """
    Genera un mapa aleatoreo.

    Returns:
        NDArray[np.int32]: El mapa generado como array 15x15 de Numpy.
    """
    total = MAP_SHAPE[0] * MAP_SHAPE[1] - 2
    counts = {val: rng.integers(lo, hi + 1) for val, (lo, hi) in CONSTRAINTS.items()}
    flat = np.zeros(total, dtype=int)
    items = np.repeat(list(counts.keys()), list(counts.values()))

    flat[: len(items)] = items
    rng.shuffle(flat)
    flat = np.concatenate(([Elemento.INICIO], flat, [Elemento.META]))

    return flat.reshape(MAP_SHAPE)


def fancy_print(mapa: NDArray[np.int32]):
    """
    Imprime el mapa con la representación que se muestra en el documento.

    Args:
        mapa: Array 2D de Numpy que representa el mapa
    """
    for row in mapa:
        for element in row:
            print(FANCY_ELEMENTOS[Elemento(element)], end=" ")
        print()


def get_close_elements(
    mapa: NDArray[np.int32], element: Elemento, target: Elemento, distance: int
) -> int:
    """
    Calcula si el elemento esta cerca de otro.

    Args:
        mapa: Array 2D de Numpy que representa el mapa.
        element: Elemento base para comprobar la proximidad.
        target: Elemento objetivo.
        distance: Distancia entre elementos.

    Returns:
        int: Regresa la cantidad de veces un elemento se encuentra cerca de otro.
    """
    pos_a = np.argwhere(mapa == element)
    pos_b = np.argwhere(mapa == target)
    same = element == target

    count = 0
    for i, (ra, ca) in enumerate(pos_a):
        others = pos_b[i + 1 :] if same else pos_b

        if len(others) == 0:
            continue
        #penaliza las distancias que sean menores a 2 distancia minima por el maestro
        dist = np.max(np.abs(others - [ra, ca]), axis=1)
        count += int(np.count_nonzero(dist < distance))
    return count


def calculate_enemy_penalty(mapa: NDArray[np.int32]) -> int:
    """
    Calcula la penalización de enemigos, lee el documento para más información.

    Args:
        mapa: Array 2D de Numpy que representa el mapa.

    Returns:
        int: Penalización de enemigos.
    """
    close_encounters_of_the_enemy_kind = get_close_elements(
        mapa, Elemento.ENEMIGO, Elemento.ENEMIGO, 2
    )
    close_to_start = get_close_elements(mapa, Elemento.ENEMIGO, Elemento.INICIO, 3)
    return close_encounters_of_the_enemy_kind + close_to_start


def count_accesible_rewards(mapa: NDArray[np.int32]) -> int:
    #Cuenta las recompensas a las que no se puede llegar desde el inicio, contando las encerradas en recuadros grandes
    queue = deque([(0, 0)])
    visited = {(0, 0)}
    count = 0

    while queue:
        row, col= queue.popleft()
        if mapa[row, col] == Elemento.RECOMPENZA:
            count += 1

        for nr, nc in [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]:
            if (
                0 <= nr < MAP_SHAPE[0]
                and 0 <= nc < MAP_SHAPE[1]
                and (nr, nc) not in visited
                and mapa[nr, nc] != Elemento.OBSTACULO
            ):
                visited.add((nr, nc))
                queue.append((nr, nc))

    return count


def calculate_reward_penalty(mapa: NDArray[np.int32]) -> int:
    """
    Calcula la penalización de recompensas, lee el documento para más información.

    Args:
        mapa: Array 2D de Numpy que representa el mapa.

    Returns:
        int: Penalización de recompensas.
    """
    #penaliza con 100 cada recompenza que no se puede alcanzar, de entender mal correegir
    total = np.count_nonzero(mapa == Elemento.RECOMPENZA)
    return int(100 * (total - count_accesible_rewards(mapa)))


def calculate_distribution_penalty(mapa: NDArray[np.int32]) -> int:
    #Penaliza zonas de 5x5 con demasiados elementos para evitar que haya mucha concentracion
    tolerancias = {
        Elemento.OBSTACULO: 2,
        Elemento.ENEMIGO: 1,
        Elemento.RECOMPENZA: 1,
    }
    penalizacion = 0

    for elemento, tolerancia in tolerancias.items():
        promedio = round(np.count_nonzero(mapa == elemento) / 9)
        for fila in range(0, MAP_SHAPE[0], 5):
            for columna in range(0, MAP_SHAPE[1], 5):
                zona = mapa[fila : fila + 5, columna : columna +5 ]
                cantidad = np.count_nonzero(zona == elemento)
                #suma solo el exceso fuera de la tolerancia
                penalizacion += max(0, abs(cantidad - promedio) - tolerancia)
    return int(penalizacion)


def calculate_path_penalty(
    mapa: NDArray[np.int32], solution: list[tuple[int, int]] | None
) -> int:
    """
    Calcula la penalización del camino, lee el documento para más información.

    Args:
        mapa: Array 2D de Numpy que representa el mapa.
        solution: Solución directa del algoritmo de busqueda.

    Returns:
        int: Penalización del camino.
    """
    return 10000 if solution is None else 0


def get_difficulty(mapa: NDArray[np.int32], solution: list[tuple[int, int]]) -> float:
    """
    Calcula la dificultad.

    Args:
        mapa: Array 2D de Numpy que representa el mapa.
        solution: Solución valida del algoritmo de busqueda.

    Returns:
        float: difucultad
    """
    blockers = get_ajacent_blockers(mapa, solution)
    adjacent_enemies = blockers[Elemento.ENEMIGO]
    adjacent_obstacles = blockers[Elemento.OBSTACULO]
    available_rewards = np.count_nonzero(mapa == Elemento.RECOMPENZA)

    return (
        5 * adjacent_enemies
        + 0.5 * len(solution)
        + 2 * adjacent_obstacles
        - 4 * available_rewards
    )


def calculate_dificulty_penalty(
    mapa: NDArray[np.int32], solution: list[tuple[int, int]], expected_dificulty: int
) -> int:
    """
    Calcula la penalización de dificultad, lee el documento para más información.

    Args:
        mapa: Array 2D de Numpy que representa el mapa.
        solution: Solución valida del algoritmo de busqueda.
        expected_dificulty: Dificultad esperada.

    Returns:
        int: Penalización de dificultad.
    """
    return np.abs(get_difficulty(mapa, solution) - expected_dificulty)


def calculate_fitness(
    mapa: NDArray[np.int32],
    solution: list[tuple[int, int]] | None,
    expected_dificulty: int,
) -> int:
    """
    Calcula el fitness del mapa.

    Args:
        mapa: Array 2D de Numpy que representa el mapa.
        solution: Solución valida del algoritmo de busqueda.
        expected_dificulty: Dificultad esperada.

    Returns:
        int: Fitness del mapa.
    """
    w1, w2, w3, w4, w5 = (50, 10, 5, 5, 2)
    p_camino = calculate_path_penalty(mapa, solution)
    p_dificultad = (
        calculate_dificulty_penalty(mapa, solution, expected_dificulty)
        if solution is not None
        else 10000
    )  # I'm so fucking tired
    p_enemigos = calculate_enemy_penalty(mapa)
    p_recompensas = calculate_reward_penalty(mapa)
    p_distribucion = calculate_distribution_penalty(mapa)

    return (
        w1 * p_camino
        + w2 * p_dificultad
        + w3 * p_enemigos
        + w4 * p_recompensas
        + w5 * p_distribucion
    )

#Oswaldo nuevamente hace acto de presencia, esta vez sin hambre de comida, pero con hambre de exito
def reparar_cantidad_elemento(matriz_mapa, caracter_elemento, limite_minimo, limite_maximo):
    """
    Oswaldo decidio ajustar los elementos, que estan representados por caracteres ('#', 'E', 'R') 
    para que cumplan con los limites que se establecieron, Oswaldo garantiza que el inicio y la meta
    no se moveran durante este proceso, pero no garantiza que el precio de la tortilla deje de subir

    El poder de Oswaldo es muy limitado

    """
    coordenadas_del_elemento = []
    coordenadas_vacias = []

    # Oswaldo decide escanear el mapa y guardar las coordenadas de todos los elementos y espacios vacios
    for fila in range(len(matriz_mapa)):
        for columna in range(len(matriz_mapa[0])):
            if matriz_mapa[fila][columna] == caracter_elemento:
                coordenadas_del_elemento.append((fila, columna))
            elif matriz_mapa[fila][columna] == '.':
                coordenadas_vacias.append((fila, columna))
                
    cantidad_actual = len(coordenadas_del_elemento)
    
    # A oswaldo no le gustan los excesos, asi que si los hay, convertimos elementos sobrantes 
    # en espacios vacíos al azar
    while cantidad_actual > limite_maximo and coordenadas_del_elemento:
        fila_elegida, columna_elegida = random.choice(coordenadas_del_elemento)
        matriz_mapa[fila_elegida][columna_elegida] = '.'
        coordenadas_del_elemento.remove((fila_elegida, columna_elegida))
        coordenadas_vacias.append((fila_elegida, columna_elegida))
        cantidad_actual -= 1
        
    # Si hay algo que oswaldo odie mas que los excesos, es la escasez, y si encontramos una,
    # tomamos espacios vacíos al azar y agregamos el elemento

    while cantidad_actual < limite_minimo and coordenadas_vacias:
        fila_elegida, columna_elegida = random.choice(coordenadas_vacias)
        matriz_mapa[fila_elegida][columna_elegida] = caracter_elemento
        coordenadas_vacias.remove((fila_elegida, columna_elegida))
        coordenadas_del_elemento.append((fila_elegida, columna_elegida))
        cantidad_actual += 1
        
    return matriz_mapa

def reparar_mapa_completo(matriz_mapa):
    """
    Oswaldo es un hombre que necesita seguridad, por lo tanto, decidio implementar esta seccion de codigo 
    para asegurarse de que el mapa cumpla con los limites establecidos para cada tipo de elemento,
    y asi evitar que el mapa sea demasiado facil o demasiado dificil

    A Oswaldo le haria bien un baño ahora, pero no tiene tiempo para eso

    """
    MINIMO_OBSTACULOS, MAXIMO_OBSTACULOS = 30, 45
    MINIMO_ENEMIGOS, MAXIMO_ENEMIGOS = 4, 8
    MINIMO_RECOMPENSAS, MAXIMO_RECOMPENSAS = 3, 6

    matriz_mapa = reparar_cantidad_elemento(matriz_mapa, '#', MINIMO_OBSTACULOS, MAXIMO_OBSTACULOS)
    matriz_mapa = reparar_cantidad_elemento(matriz_mapa, 'E', MINIMO_ENEMIGOS, MAXIMO_ENEMIGOS)
    matriz_mapa = reparar_cantidad_elemento(matriz_mapa, 'R', MINIMO_RECOMPENSAS, MAXIMO_RECOMPENSAS)
    
    return matriz_mapa

def generar_y_guardar_mapas(cantidad=100, nombre_archivo="poblacion_inicial.json"):
    """
    Oswaldo preparo la tierra prometida. 
    guardara la peregrinacion inicial en un archivo Json (quien es Json?)
    en la carpeta 'Mapas' del proyecto
    """
    # 1. Oswaldo necesita saber la ruta actual del generador 
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Oswaldo entra a la carpeta de los mapas, QUE HORROR, parece que no ha pasado una escoba
    # por aqui en años
    directorio_raiz = os.path.dirname(directorio_actual)
    carpeta_mapas = os.path.join(directorio_raiz, "Mapas")
    
    # 3. Oswaldo piensa que algo a prueba de estupidos es algo bien hecho, asi que 
    # para salvarse de si mismo, pone esto por si borro la carpeta
    os.makedirs(carpeta_mapas, exist_ok=True)
    
    # 4. Oswaldo define la ruta del archivo, pero que, parece que oswaldo ya no puede mas
    # oswaldo quiere tomar un baño e irse a dormir
    ruta_completa = os.path.join(carpeta_mapas, nombre_archivo)
    
    lista_de_mapas = []
    
    for _ in range(cantidad):
        # Queda poco oswaldo, se fuerte, guardamos el mapa
        mapa_np = generate_random_map()
        
        # Oswaldo le da una pasada de control de calidad, oswaldo no sabe que debe de buscar
        mapa_reparado = reparar_mapa_completo(mapa_np)
        
        # Json no entiende numpy, asi que lo pasamos a lista, jaja, estupido Json
        mapa_lista = mapa_reparado.tolist()
        
        lista_de_mapas.append({
            "matrix": mapa_lista
        })
        
    # 5. Guardar el archivo JSON, pero en fa que ya nos queremos ir a dormi
    with open(ruta_completa, 'w', encoding='utf-8') as archivo:
        json.dump(lista_de_mapas, archivo)
        
    print(f"Oswaldo ha guardado {cantidad} mapas exitosamente en:\n{ruta_completa}")
#Fin del trabajo de oswaldo, por ahora, oswaldo esta roto por dentro, pero su sueño de jugar al zelda remake 
#sigue en pie