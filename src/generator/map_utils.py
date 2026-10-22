from collections import deque
from typing import Annotated

import numpy as np
from numpy._typing import NDArray

from path_search.utils import get_ajacent_blockers
from representation import BLOCKERS, MAP_SHAPE, Elemento

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


def calculate_distrubution_penalty(mapa: NDArray[np.int32]) -> int:
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


def get_dificulty(mapa: NDArray[np.int32], solution: list[tuple[int, int]]) -> float:
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
    return np.abs(get_dificulty(mapa, solution) - expected_dificulty)


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
    p_distribucion = calculate_distrubution_penalty(mapa)

    return (
        w1 * p_camino
        + w2 * p_dificultad
        + w3 * p_enemigos
        + w4 * p_recompensas
        + w5 * p_distribucion
    )
