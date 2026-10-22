import math
import random

import numpy as np
from numpy.typing import NDArray

from generator.map_utils import CONSTRAINTS, calculate_fitness
from path_search.breadth import bfs
from representation import MAP_SHAPE, Elemento

rng = np.random.default_rng()

MOVABLE_MASK = np.ones(MAP_SHAPE, dtype=bool)
MOVABLE_MASK[0, 0] = False
MOVABLE_MASK[14, 14] = False


def random_cell(mapa: NDArray[np.int32], value: Elemento):
    mask = MOVABLE_MASK & (mapa == value)
    cells = np.argwhere(mask)
    #devolvera none si no hay casilla disponible para el valor buscado
    if len(cells) == 0:
        return None
    return tuple(cells[rng.integers(len(cells))])


def swap_move(candidate: NDArray[np.int32]):
    #intercambia una casilla libre con una ocupada
    first = random_cell(candidate, Elemento.TRANSITABLE)
    occupied = np.argwhere(MOVABLE_MASK & (candidate != Elemento.TRANSITABLE))
    if first is None or len(occupied) == 0:
        return

    second = tuple(occupied[rng.integers(len(occupied))])
    candidate[first], candidate[second] = candidate[second], candidate[first]


def count_move(candidate: NDArray[np.int32]):
    element = random.choice(list(CONSTRAINTS))
    count = np.count_nonzero(candidate == element)
    #Permite agregar como quitar elementos sin salir de los rangos en lugar de favorecer la adición
    lo, hi = CONSTRAINTS[element]
    if count < lo or (count < hi and random.random() < 0.5):
        cell = random_cell(candidate, Elemento.TRANSITABLE)
        if cell is not None:
            candidate[cell] = element
    elif count > lo:
        cell = random_cell(candidate, element)
        if cell is not None:
            candidate[cell] = Elemento.TRANSITABLE


def neighbor(mapa: NDArray[np.int32]) -> NDArray[np.int32]:
    candidate = mapa.copy()
    count_move(candidate)
    if random.random() < 0.5:
        swap_move(candidate)
    return candidate


def simulated_annealing(
    mapa: NDArray[np.int32],
    expected_dificulty: int,
    steps: int = 10000,
    temp: float = 50,
    cooling_rate: float = 0.999,
) -> NDArray[np.int32]:
    current = mapa.copy()
    current_solution = bfs(current)
    current_score = calculate_fitness(current, current_solution, expected_dificulty)

    best = current.copy()
    best_score = current_score

    for _ in range(steps):
        candidate = neighbor(current)
        solution = bfs(candidate)
        score = calculate_fitness(candidate, solution, expected_dificulty)
        delta = score - current_score
        if delta <= 0 or random.random() < math.exp(-delta / temp):
            current = candidate
            current_score = score

            if current_score < best_score:
                best = current.copy()
                best_score = current_score
        temp *= cooling_rate
    return best
