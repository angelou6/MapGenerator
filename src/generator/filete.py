import math
import random

import numpy as np
from numpy.typing import NDArray

from generator.map_utils import CONSTRAINTS,DEFAULT_WEIGHTS, calculate_fitness
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


def count_move(
    candidate: NDArray[np.int32],
    rangos: dict[Elemento, tuple[int, int]] = CONSTRAINTS,
):
    #usa los limites del escenario elegido para meter o quitar elementos, y si no se especifica, pues deja los normales
    element = random.choice(list(rangos))
    count = np.count_nonzero(candidate == element)
    lo, hi = rangos[element]

#Permite agregar como quitar elementos sin salir de los rangos en lugar de favorecer la adición
    if count < lo or (count < hi and random.random() < 0.5):
        cell = random_cell(candidate, Elemento.TRANSITABLE)
        if cell is not None:
            candidate[cell] = element
    elif count > lo:
        cell = random_cell(candidate, element)
        if cell is not None:
            candidate[cell] = Elemento.TRANSITABLE


def neighbor(
    mapa: NDArray[np.int32],
    rangos: dict[Elemento, tuple[int, int]] = CONSTRAINTS,
) -> NDArray[np.int32]:
    candidate = mapa.copy()
    #envia los rangos seleccionados al movimiento veciono
    count_move(candidate, rangos)
    if random.random() < 0.5:
        swap_move(candidate)
    return candidate


def simulated_annealing(
    mapa: NDArray[np.int32],
    expected_dificulty: int,
    steps: int = 10000,
    temp: float = 50,
    cooling_rate: float = 0.999,
    history: list[tuple[float, float]] | None = None,
    weights: tuple[float, float, float, float, float] = DEFAULT_WEIGHTS,
    rangos: dict[Elemento, tuple[int, int]] = CONSTRAINTS,
) -> NDArray[np.int32]:
    current = mapa.copy()
    current_solution = bfs(current)
    #Evalua con los pesos elegidos desde la ejecucion
    current_score = calculate_fitness(current, current_solution, expected_dificulty, weights)

    best = current.copy()
    best_score = current_score

    #Guarda el fitness actual y el mejor encontrado 
    if history is not None:
        history.clear()
        history.append((float(current_score), float(best_score)))

    for _ in range(steps):
        #los vecionos respetaran los rangos del escenario elegido
        candidate = neighbor(current, rangos)
        solution = bfs(candidate)
        #usa los mismo pesos que todos los candidatos
        score = calculate_fitness(candidate, solution, expected_dificulty, weights)
        delta = score - current_score
        if delta <= 0 or random.random() < math.exp(-delta / temp):
            current = candidate
            current_score = score

            if current_score < best_score:
                best = current.copy()
                best_score = current_score
        temp *= cooling_rate

        #Registran los valores despues de cada iteracion
        if history is not None:
            history.append((float(current_score), float(best_score)))

    return best
