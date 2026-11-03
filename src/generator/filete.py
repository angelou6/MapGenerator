import math

import numpy as np
from numpy.typing import NDArray

from generator.map_utils import (
    CONSTRAINTS,
    DEFAULT_WEIGHTS,
    MOVABLE_MASK,
    Weights,
    evaluate,
    generate_random_map,
)
from generator.scenarios import get_scenario
from representation import Elemento

rng = np.random.default_rng()


def random_cell(
    mapa: NDArray[np.int32], value: Elemento | None = None
) -> tuple[int, int] | None:
    """
    Elige una celda movible al azar que contenga el elemento indicado.

    Si value es None, elige cualquier celda movible que no sea transitable.

    Args:
        mapa: Array 2D de Numpy que representa el mapa.
        value: Elemento a buscar, o None para cualquier celda ocupada.

    Returns:
        tuple[int, int] | None: Coordenadas (fila, columna) de la celda elegida,
        o None si no hay casillas disponibles.
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
    """
    Intercambia la posición de un elemento ocupado con una celda libre transitable.

    Args:
        candidate: Array 2D de Numpy que representa el mapa. Se modifica en el lugar.
    """
    first = random_cell(candidate)
    second = random_cell(candidate, Elemento.TRANSITABLE)
    if first is None or second is None:
        return
    candidate[first], candidate[second] = candidate[second], candidate[first]


def count_move(
    candidate: NDArray[np.int32],
    rangos: dict[Elemento, tuple[int, int]] | None = None,
) -> None:
    """
    Agrega o elimina un elemento al azar asegurando que se respeten los rangos válidos.

    Args:
        candidate: Array 2D de Numpy que representa el mapa. Se modifica en el lugar.
        rangos: Diccionario con los límites permitidos para cada tipo de elemento. Si
            es None, se utilizan los rangos predeterminados de CONSTRAINTS.
    """
    if rangos is None:
        rangos = CONSTRAINTS
    elements = list(rangos)
    element = elements[rng.integers(len(elements))]
    count = np.count_nonzero(candidate == element)
    lo, hi = rangos[element]

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


def neighbor(
    mapa: NDArray[np.int32],
    rangos: dict[Elemento, tuple[int, int]] | None = None,
) -> NDArray[np.int32]:
    """
    Genera un estado vecino aplicando una perturbación aleatoria al mapa.

    Args:
        mapa: Array 2D de Numpy que representa el mapa actual.
        rangos: Diccionario opcional con los rangos permitidos para cada elemento.

    Returns:
        NDArray[np.int32]: Nuevo mapa generado como vecino.
    """
    candidate = mapa.copy()
    if rng.random() < 0.7:
        swap_move(candidate)
    else:
        count_move(candidate, rangos)
    return candidate


def simulated_annealing(
    expected_dificulty: int,
    steps: int = 5000,
    temperature: float = 50,
    cooling_rate: float = 0.999,
    escenario: str | None = None,
    weights: Weights = DEFAULT_WEIGHTS,
    history: dict[str, list[float]] | None = None,
) -> NDArray[np.int32]:
    """
    Optimiza un mapa mediante el algoritmo de Recocido Simulado.

    Args:
        expected_dificulty: Dificultad objetivo esperada para el mapa.
        steps: Número máximo de pasos o iteraciones de enfriamiento.
        temperature: Temperatura inicial del sistema, calibrada para DEFAULT_WEIGHTS.
            Se escala proporcionalmente a la suma de los pesos para que la
            probabilidad de aceptación no dependa de la magnitud de los pesos.
        cooling_rate: Factor de enfriamiento geométrico aplicado en cada paso.
        escenario: Nombre opcional del escenario para ajustar las restricciones y dificultad.
        weights: Pesos del fitness. Por defecto usa DEFAULT_WEIGHTS.
        history: Diccionario opcional donde se guarda, por paso, el mejor fitness,
            el fitness actual y la temperatura.

    Returns:
        NDArray[np.int32]: El mejor mapa encontrado durante la optimización.
    """
    config = get_scenario(escenario)
    rangos = config["rangos"]
    camino_largo = config["camino_largo"]
    temperature *= sum(weights) / sum(DEFAULT_WEIGHTS)

    current = generate_random_map(escenario)
    current_score = evaluate(current, expected_dificulty, weights, camino_largo)

    best = current.copy()
    best_score = current_score

    for _ in range(steps):
        candidate = neighbor(current, rangos)
        score = evaluate(candidate, expected_dificulty, weights, camino_largo)

        delta = score - current_score
        if delta <= 0 or (
            temperature > 0 and rng.random() < math.exp(-delta / temperature)
        ):
            current = candidate
            current_score = score

            if score < best_score:
                best = candidate.copy()
                best_score = score
        if history is not None:
            history.setdefault("mejor fitness", []).append(best_score)
            history.setdefault("fitness actual", []).append(current_score)
            history.setdefault("temperatura", []).append(temperature)
        temperature *= cooling_rate
    return best
