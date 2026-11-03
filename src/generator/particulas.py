import numpy as np
from numpy.typing import NDArray

from generator.map_utils import evaluate, repair
from representation import MAP_SHAPE

rng = np.random.default_rng()

# Limites de los intervalos: [0, 0.70) libre, [0.70, 0.85) obstaculo,
# [0.85, 0.95) enemigo y [0.95, 1] recompensa.
THRESHOLDS = (0.70, 0.85, 0.95)


def discretize(position: NDArray[np.float64]) -> NDArray[np.int32]:
    """
    Convierte la posicion continua de una particula en un mapa reparado.

    La posicion se usa como prioridad en la reparacion: si sobran elementos se
    quitan los de valor mas bajo y si faltan se ponen donde el valor es mas alto.
    """
    mapa = np.digitize(position, THRESHOLDS).astype(np.int32)
    return repair(mapa, position)


def particle_swarm(
    expected_dificulty: int,
    particles: int = 30,
    iterations: int = 300,
    inertia: float = 0.7,
    c1: float = 1.5,
    c2: float = 1.5,
    max_velocity: float = 0.2,
) -> NDArray[np.int32]:
    shape = (particles, *MAP_SHAPE)
    positions = rng.random(shape)
    velocities = rng.uniform(-max_velocity, max_velocity, shape)

    maps = [discretize(position) for position in positions]
    scores = np.array([evaluate(mapa, expected_dificulty) for mapa in maps])

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

        for i, position in enumerate(positions):
            mapa = discretize(position)
            score = evaluate(mapa, expected_dificulty)

            if score < pbest_scores[i]:
                pbest[i] = position
                pbest_scores[i] = score

                if score < best_score:
                    gbest = position.copy()
                    best = mapa.copy()
                    best_score = score
    return best
