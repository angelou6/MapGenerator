import math

import numpy as np
from numpy.typing import NDArray

from generator.map_utils import (
    CONSTRAINTS,
    MOVABLE_MASK,
    evaluate,
    generate_random_map,
)
from representation import Elemento

rng = np.random.default_rng()


def random_cell(
    mapa: NDArray[np.int32], value: Elemento | None = None
) -> tuple[int, int] | None:
    """
    Elige una celda movible al azar que contenga `value`.
    Si `value` es None, elige cualquier celda movible que no sea transitable.
    """
    if value is None:
        mask = MOVABLE_MASK & (mapa != Elemento.TRANSITABLE)
    else:
        mask = MOVABLE_MASK & (mapa == value)
    cells = np.argwhere(mask)
    if len(cells) == 0:
        return None
    row, col = cells[rng.integers(len(cells))]
    return int(row), int(col)


def swap_move(candidate: NDArray[np.int32]) -> None:
    """Mueve un elemento (obstaculo, enemigo o recompensa) a una celda libre."""
    first = random_cell(candidate)
    second = random_cell(candidate, Elemento.TRANSITABLE)
    if first is None or second is None:
        return
    candidate[first], candidate[second] = candidate[second], candidate[first]


def count_move(candidate: NDArray[np.int32]) -> None:
    """Agrega o quita un elemento sin salir de los rangos de CONSTRAINTS."""
    elements = list(CONSTRAINTS)
    element = elements[rng.integers(len(elements))]
    count = np.count_nonzero(candidate == element)
    lo, hi = CONSTRAINTS[element]

    can_add = count < hi
    can_remove = count > lo
    if not (can_add or can_remove):
        return

    if can_add and (not can_remove or rng.random() < 0.5):
        cell = random_cell(candidate, Elemento.TRANSITABLE)
        if cell is not None:
            candidate[cell] = element
    else:
        cell = random_cell(candidate, element)
        if cell is not None:
            candidate[cell] = Elemento.TRANSITABLE


def neighbor(mapa: NDArray[np.int32]) -> NDArray[np.int32]:
    """Genera un vecino aplicando un solo movimiento pequeño al mapa."""
    candidate = mapa.copy()
    if rng.random() < 0.7:
        swap_move(candidate)
    else:
        count_move(candidate)
    return candidate


def simulated_annealing(
    expected_dificulty: int,
    steps: int = 10000,
    temperature: float = 50,
    cooling_rate: float = 0.999,
) -> NDArray[np.int32]:
    current = generate_random_map()
    current_score = evaluate(current, expected_dificulty)

    best = current.copy()
    best_score = current_score

    for _ in range(steps):
        candidate = neighbor(current)
        score = evaluate(candidate, expected_dificulty)

        delta = score - current_score
        if delta <= 0 or rng.random() < math.exp(-delta / temperature):
            current = candidate
            current_score = score

            if score < best_score:
                best = candidate.copy()
                best_score = score
        temperature *= cooling_rate
    return best
