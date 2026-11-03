from collections.abc import Sequence
from itertools import combinations
from typing import Annotated

import numpy as np
from numpy._typing import NDArray

from generator.scenarios import DEFAULT_CONSTRAINTS, get_scenario
from path_search.breadth import bfs, reachable_tiles
from path_search.utils import get_ajacent_blockers
from representation import MAP_SHAPE, Elemento

rng = np.random.default_rng()

CONSTRAINTS: Annotated[
    dict[Elemento, tuple[int, int]], "Rango de objetos posibles dentro del mapa."
] = DEFAULT_CONSTRAINTS

MOVABLE_MASK = np.ones(MAP_SHAPE, dtype=bool)
MOVABLE_MASK[0, 0] = False
MOVABLE_MASK[-1, -1] = False

FANCY_ELEMENTOS = {
    Elemento.TRANSITABLE: ".",
    Elemento.OBSTACULO: "#",
    Elemento.ENEMIGO: "E",
    Elemento.RECOMPENZA: "R",
    Elemento.INICIO: "I",
    Elemento.META: "M",
}


def generate_random_map(escenario: str | None = None) -> NDArray[np.int32]:
    """
    Genera un mapa aleatorio.

    Args:
        escenario: Nombre opcional del escenario para ajustar las restricciones.

    Returns:
        NDArray[np.int32]: El mapa generado como array 15x15 de Numpy.
    """
    rangos: dict[Elemento, tuple[int, int]] = CONSTRAINTS
    if escenario is not None:
        rangos = get_scenario(escenario)["rangos"]

    total = MAP_SHAPE[0] * MAP_SHAPE[1] - 2
    counts = {val: rng.integers(lo, hi + 1) for val, (lo, hi) in rangos.items()}
    flat = np.zeros(total, dtype=int)
    items = np.repeat(list(counts.keys()), list(counts.values()))

    flat[: len(items)] = items
    rng.shuffle(flat)
    flat = np.concatenate(([Elemento.INICIO], flat, [Elemento.META]))

    return flat.reshape(MAP_SHAPE)


def repair(
    mapa: NDArray[np.int32],
    priority: NDArray[np.float64] | None = None,
    rangos: dict[Elemento, tuple[int, int]] | None = None,
) -> NDArray[np.int32]:
    """
    Repara el mapa para que respete los limites de los rangos y que inicio y
    meta esten en su lugar.

    Si sobran elementos se quitan los de menor prioridad; si faltan se agregan
    en las celdas transitables de mayor prioridad. Sin prioridad se elige al azar.

    Args:
        mapa: Array 2D de Numpy que representa el mapa. Se modifica en el lugar.
        priority: Array 2D con la prioridad de cada celda.
        rangos: Rangos de elementos a respetar. Si es None se usa CONSTRAINTS.

    Returns:
        NDArray[np.int32]: El mismo mapa ya reparado.
    """
    if priority is None:
        priority = rng.random(MAP_SHAPE)
    if rangos is None:
        rangos = CONSTRAINTS

    mapa[~MOVABLE_MASK] = Elemento.TRANSITABLE
    mapa[mapa > Elemento.RECOMPENZA] = Elemento.TRANSITABLE

    for element, (lo, hi) in rangos.items():
        cells = np.argwhere(mapa == element)
        if len(cells) > hi:
            order = np.argsort(priority[cells[:, 0], cells[:, 1]])
            extra = cells[order[: len(cells) - hi]]
            mapa[extra[:, 0], extra[:, 1]] = Elemento.TRANSITABLE
        elif len(cells) < lo:
            free = np.argwhere(MOVABLE_MASK & (mapa == Elemento.TRANSITABLE))
            order = np.argsort(-priority[free[:, 0], free[:, 1]])
            chosen = free[order[: lo - len(cells)]]
            mapa[chosen[:, 0], chosen[:, 1]] = element

    mapa[0, 0] = Elemento.INICIO
    mapa[-1, -1] = Elemento.META
    return mapa


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

        dist = np.max(np.abs(others - [ra, ca]), axis=1)
        if np.any(dist <= distance):
            count += 1
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


def calculate_reward_penalty(mapa: NDArray[np.int32]) -> int:
    """
    Calcula la penalización de recompensas, lee el documento para más información.

    Cada recompensa que no se puede alcanzar desde el inicio (usando BFS)
    recibe una penalización de 100.

    Args:
        mapa: Array 2D de Numpy que representa el mapa.

    Returns:
        int: Penalización de recompensas.
    """
    reachable = reachable_tiles(mapa)
    rewards = map(tuple, np.argwhere(mapa == Elemento.RECOMPENZA).tolist())
    unreachable = sum(1 for reward in rewards if reward not in reachable)
    return unreachable * 100


def calculate_distrubution_penalty(mapa: NDArray[np.int32]) -> int:
    """
    Calcula la penalización de distribución.

    Divide el mapa en zonas de 5x5 y penaliza las zonas cuya cantidad de
    obstáculos, enemigos o recompensas se aleja demasiado del promedio.

    Args:
        mapa: Array 2D de Numpy que representa el mapa.

    Returns:
        int: Penalización de distribución.
    """
    tolerancias = {
        Elemento.OBSTACULO: 2,
        Elemento.ENEMIGO: 1,
        Elemento.RECOMPENZA: 1,
    }
    zone = 5
    rows, cols = MAP_SHAPE
    n_zones = (rows // zone) * (cols // zone)

    penalizacion = 0
    for elemento, tolerancia in tolerancias.items():
        promedio = round(np.count_nonzero(mapa == elemento) / n_zones)
        for fila in range(0, rows, zone):
            for columna in range(0, cols, zone):
                zona = mapa[fila : fila + zone, columna : columna + zone]
                cantidad = np.count_nonzero(zona == elemento)
                penalizacion += max(0, abs(cantidad - promedio) - tolerancia)
    return int(penalizacion)


def calculate_path_penalty(solution: list[tuple[int, int]] | None) -> int:
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
    w1, w2, w3, w4, w5 = (50, 10, 3, 5, 2)
    p_camino = calculate_path_penalty(solution)
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


def evaluate(mapa: NDArray[np.int32], expected_dificulty: int) -> float:
    """
    Busca el camino del mapa con BFS y calcula su fitness.

    Args:
        mapa: Array 2D de Numpy que representa el mapa.
        expected_dificulty: Dificultad esperada.

    Returns:
        float: Fitness del mapa.
    """
    return calculate_fitness(mapa, bfs(mapa), expected_dificulty)


def calculate_hamming_distance(
    mapa_a: NDArray[np.int32], mapa_b: NDArray[np.int32]
) -> int:
    """
    Cuenta las casillas que son diferentes entre dos mapas (distancia de Hamming).

    Args:
        mapa_a: Array 2D de Numpy que representa el primer mapa.
        mapa_b: Array 2D de Numpy que representa el segundo mapa.

    Returns:
        int: Número de casillas en las que ambos mapas difieren.

    Raises:
        ValueError: Si los mapas no tienen el mismo tamaño.
    """
    if mapa_a.shape != mapa_b.shape:
        raise ValueError("Los mapas deben tener el mismo tamaño")

    return int(np.count_nonzero(mapa_a != mapa_b))


def calculate_average_diversity(mapas: Sequence[NDArray[np.int32]]) -> float:
    """
    Calcula la distancia de Hamming promedio entre todos los pares de mapas.

    Sirve para medir qué tan distintos son los mapas generados (por ejemplo, la
    población de un algoritmo o varias ejecuciones), sin modificar el fitness.

    Args:
        mapas: Secuencia de mapas a comparar.

    Returns:
        float: Promedio de casillas diferentes por par de mapas, o 0.0 si hay
        menos de dos mapas.
    """
    distancias = [calculate_hamming_distance(a, b) for a, b in combinations(mapas, 2)]
    if not distancias:
        return 0.0
    return float(np.mean(distancias))
