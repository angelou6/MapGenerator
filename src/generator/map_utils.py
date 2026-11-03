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

#pesos predeterminados para el fitness. Se cambian al llamar la funcion sin tener que editar estos valores
DEFAULT_WEIGHTS = (50, 10, 5, 5, 2)

FANCY_ELEMENTOS = {
    Elemento.TRANSITABLE: ".",
    Elemento.OBSTACULO: "#",
    Elemento.ENEMIGO: "E",
    Elemento.RECOMPENZA: "R",
    Elemento.INICIO: "I",
    Elemento.META: "M",
}


def generate_random_map(
    escenario: str | dict[Elemento, tuple[int, int]] | None = None,
) -> NDArray[np.int32]:
    """
    Genera un mapa aleatoreo.

    Args:
        escenario: Nombre de un escenario (ver scenarios.py), un diccionario de
            rangos {Elemento: (min, max)} (presets personalizados) o None para
            usar los rangos normales del documento.

    Returns:
        NDArray[np.int32]: El mapa generado como array 15x15 de Numpy.
    """
    #deja elegir los rangos del escenario sin modificar los rangos originales
    rangos = CONSTRAINTS
    if isinstance(escenario, dict):
        rangos = escenario
    elif escenario is not None:
        from generator.scenarios import get_scenario

        rangos = get_scenario(escenario)["rangos"]

    total = MAP_SHAPE[0] * MAP_SHAPE[1] - 2
    counts = {val: rng.integers(lo, hi + 1) for val, (lo, hi) in rangos.items()}
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
    #el documento define R como recompensas accesibles (no las encerradas)
    available_rewards = count_accesible_rewards(mapa)

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
    weights: tuple[float, float, float, float, float] = DEFAULT_WEIGHTS,
) -> float:
    """
    Calcula el fitness del mapa.

    Args:
        mapa: Array 2D de Numpy que representa el mapa.
        solution: Solución valida del algoritmo de busqueda.
        expected_dificulty: Dificultad esperada.

    Returns:
        float: Fitness del mapa.
    """
    #utiliza los pesos recibidos para calcular el fitness. Si no se indican utilizara los default de arriba
    w1, w2, w3, w4, w5 = (weights)
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

def reparar_cantidad_elemento(matriz_mapa, elemento, limite_minimo, limite_maximo):
    """
    Ajusta (en el mismo arreglo) la cantidad de un elemento para que quede dentro de
    [limite_minimo, limite_maximo]. Si sobran, convierte algunos al azar en espacio
    transitable; si faltan, coloca el elemento en espacios transitables al azar.
    El inicio (0,0) y la meta (14,14) nunca se tocan.

    Args:
        matriz_mapa: Mapa como array 2D de Numpy con los codigos enteros de Elemento.
        elemento: Codigo del elemento a reparar (Elemento.OBSTACULO, etc.).
        limite_minimo: Cantidad minima permitida.
        limite_maximo: Cantidad maxima permitida.
    """
    elemento = int(elemento)
    movible = np.ones(MAP_SHAPE, dtype=bool)
    movible[0, 0] = False
    movible[-1, -1] = False

    posiciones = np.argwhere(movible & (matriz_mapa == elemento))
    exceso = len(posiciones) - limite_maximo
    if exceso > 0:
        for idx in rng.choice(len(posiciones), size=exceso, replace=False):
            matriz_mapa[tuple(posiciones[idx])] = Elemento.TRANSITABLE
        return matriz_mapa

    faltante = limite_minimo - len(posiciones)
    if faltante > 0:
        libres = np.argwhere(movible & (matriz_mapa == Elemento.TRANSITABLE))
        faltante = min(faltante, len(libres))
        if faltante > 0:
            for idx in rng.choice(len(libres), size=faltante, replace=False):
                matriz_mapa[tuple(libres[idx])] = elemento
    return matriz_mapa


def reparar_mapa_completo(matriz_mapa, rangos=None):
    """
    Repara el mapa para que obstaculos, enemigos y recompensas respeten los rangos
    indicados (por defecto los del documento, CONSTRAINTS). Modifica y regresa el mapa.
    """
    rangos = CONSTRAINTS if rangos is None else rangos
    for elemento, (minimo, maximo) in rangos.items():
        reparar_cantidad_elemento(matriz_mapa, elemento, minimo, maximo)
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

#mide la diversidad sin modificar el fitness ni los mapas
def calculate_hamming_distance(
        mapa_a: NDArray[np.int32], mapa_b: NDArray[np.int32]
) -> int:
    """Cuenta las casillas que son diferentes entre dos mapas"""
    #ccomprueba que ambos mapas tengan el mismo tamaño
    if mapa_a.shape != mapa_b.shape:
        raise ValueError("Los mapas deben tener el mismo tamaño")

    return int(np.count_nonzero(mapa_a != mapa_b))


def calculate_average_diversity(mapas: list[NDArray[np.int32]]) -> float:
    """Calcula el promedio de diferencias entre los pares de mapas"""
    if len(mapas) < 2:
        return 0.0

    total = 0
    pares = 0
    #compara las parejas una a la vez
    for i in range(len(mapas)):
        for j in range(i + 1, len(mapas)):
            total += calculate_hamming_distance(mapas[i], mapas[j])
            pares += 1

    return total / pares


def fijar_semilla(semilla: int) -> None:
    """
    Fija la semilla de TODOS los generadores aleatorios del proyecto
    (random y los rng de numpy de map_utils, filete y particulas)
    para que una ejecucion se pueda repetir igual.
    """
    import sys

    random.seed(semilla)
    for nombre in ("generator.map_utils", "generator.filete", "generator.particulas"):
        modulo = sys.modules.get(nombre)
        if modulo is not None:
            modulo.rng = np.random.default_rng(semilla)
