import numpy as np
from numpy.typing import NDArray

from generator.map_utils import (
    DEFAULT_WEIGHTS,
    Weights,
    calculate_average_diversity,
    evaluate,
    repair,
)
from generator.scenarios import get_scenario
from representation import MAP_SHAPE, Elemento

rng = np.random.default_rng()

# Límites de los intervalos: [0, 0.70) libre, [0.70, 0.85) obstáculo,
# [0.85, 0.95) enemigo y [0.95, 1] recompensa.
THRESHOLDS = (0.70, 0.85, 0.95)


def discretize(
    position: NDArray[np.float64],
    rangos: dict[Elemento, tuple[int, int]] | None = None,
) -> NDArray[np.int32]:
    """
    Convierte la posición continua de una partícula en un mapa discreto reparado.

    La posición se usa como prioridad en la reparación: si sobran elementos se
    quitan los de valor más bajo y si faltan se colocan donde el valor es más alto.

    Args:
        position: Array 2D de flotantes en [0, 1] con la posición continua de la partícula.
        rangos: Diccionario opcional con los rangos permitidos para cada elemento.

    Returns:
        NDArray[np.int32]: Mapa discreto de enteros reparado conforme a los rangos.
    """
    mapa = np.digitize(position, THRESHOLDS).astype(np.int32)
    return repair(mapa, position, rangos)


def particle_swarm(
    expected_dificulty: int,
    particles: int = 30,
    iterations: int = 300,
    inertia: float = 0.7,
    c1: float = 1.5,
    c2: float = 1.5,
    max_velocity: float = 0.2,
    escenario: str | None = None,
    weights: Weights = DEFAULT_WEIGHTS,
    history: dict[str, list[float]] | None = None,
) -> NDArray[np.int32]:
    """
    Ejecuta la optimización por enjambre de partículas para encontrar un mapa óptimo.

    Args:
        expected_dificulty: Dificultad objetivo esperada para el mapa generado.
        particles: Cantidad de partículas en el enjambre.
        iterations: Número de iteraciones o pasos de actualización del enjambre.
        inertia: Factor de inercia w sobre la velocidad anterior.
        c1: Coeficiente cognitivo de atracción hacia la mejor posición individual.
        c2: Coeficiente social de atracción hacia la mejor posición global.
        max_velocity: Límite superior de velocidad para acotar los desplazamientos.
        escenario: Nombre opcional del escenario para ajustar las restricciones y dificultad.
        weights: Pesos del fitness. Por defecto usa DEFAULT_WEIGHTS.
        history: Diccionario opcional donde se guarda, por iteración, el mejor
            fitness, el fitness mediano y la diversidad del enjambre.

    Returns:
        NDArray[np.int32]: El mejor mapa encontrado por el enjambre de partículas.
    """
    rangos = get_scenario(escenario)["rangos"] if escenario is not None else None
    shape = (particles, *MAP_SHAPE)
    positions = rng.random(shape)
    velocities = rng.uniform(-max_velocity, max_velocity, shape)

    maps = [discretize(position, rangos) for position in positions]
    scores = np.array([evaluate(mapa, expected_dificulty, weights) for mapa in maps])

    pbest = positions.copy()
    pbest_scores = scores.copy()

    best_index = int(np.argmin(scores))
    gbest = positions[best_index].copy()
    best = maps[best_index].copy()
    best_score = scores[best_index]

    for _ in range(iterations):
        r1 = rng.random(shape)
        r2 = rng.random(shape)
        velocities = (
            inertia * velocities
            + c1 * r1 * (pbest - positions)
            + c2 * r2 * (gbest - positions)
        )
        velocities = np.clip(velocities, -max_velocity, max_velocity)
        positions = np.clip(positions + velocities, 0, 1)

        maps = []
        for i, position in enumerate(positions):
            mapa = discretize(position, rangos)
            score = evaluate(mapa, expected_dificulty, weights)
            maps.append(mapa)
            scores[i] = score

            if score < pbest_scores[i]:
                pbest[i] = position
                pbest_scores[i] = score

                if score < best_score:
                    gbest = position.copy()
                    best = mapa.copy()
                    best_score = score
        if history is not None:
            history.setdefault("mejor fitness", []).append(float(best_score))
            history.setdefault("fitness mediano", []).append(float(np.median(scores)))
            history.setdefault("diversidad", []).append(
                calculate_average_diversity(maps)
            )
    return best
