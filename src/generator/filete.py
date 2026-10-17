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
    return tuple(cells[rng.integers(len(cells))])


def swap_move(candidate: NDArray[np.int32]):
    first = random_cell(candidate, Elemento.TRANSITABLE)
    second = random_cell(candidate, Elemento(candidate[first]))
    candidate[first], candidate[second] = candidate[second], candidate[first]


def count_move(candidate: NDArray[np.int32]):
    element = random.choice(list(CONSTRAINTS))
    count = np.count_nonzero(candidate == element)

    lo, hi = CONSTRAINTS[element]
    if count < hi:
        cell = random_cell(candidate, Elemento.TRANSITABLE)
        candidate[cell] = element
    elif count > lo:
        cell = random_cell(candidate, element)
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
