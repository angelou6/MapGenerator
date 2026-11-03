import numpy as np
from numpy.typing import NDArray

from generator.map_utils import (
    MOVABLE_MASK,
    evaluate,
    generate_random_map,
    repair,
)
from representation import MAP_SHAPE, Elemento

rng = np.random.default_rng()

BLOCK = 5


def tournament(
    population: list[NDArray[np.int32]], scores: list[float], size: int
) -> NDArray[np.int32]:
    """Selecciona al mejor de `size` individuos elegidos al azar."""
    contenders = rng.choice(len(population), size=size, replace=False)
    winner = int(contenders[np.argmin(np.asarray(scores)[contenders])])
    return population[winner]


def two_point_crossover(
    first: NDArray[np.int32], second: NDArray[np.int32]
) -> NDArray[np.int32]:
    """Cruza de dos puntos sobre el cromosoma de 225 genes."""
    a, b = sorted(rng.choice(first.size, size=2, replace=False))
    child = first.flatten()
    child[a:b] = second.flatten()[a:b]
    return child.reshape(MAP_SHAPE)


def block_crossover(
    first: NDArray[np.int32], second: NDArray[np.int32]
) -> NDArray[np.int32]:
    """Cruza por regiones: cada bloque de 5x5 se hereda de alguno de los padres."""
    child = first.copy()
    rows, cols = MAP_SHAPE
    for fila in range(0, rows, BLOCK):
        for columna in range(0, cols, BLOCK):
            if rng.random() < 0.5:
                region = (slice(fila, fila + BLOCK), slice(columna, columna + BLOCK))
                child[region] = second[region]
    return child


def mutate(candidate: NDArray[np.int32], mutation_rate: float) -> None:
    """Cambia el contenido de algunas celdas por un elemento al azar."""
    mask = MOVABLE_MASK & (rng.random(MAP_SHAPE) < mutation_rate)
    candidate[mask] = rng.integers(
        Elemento.TRANSITABLE, Elemento.RECOMPENZA + 1, size=np.count_nonzero(mask)
    )


def genetic_algorithm(
    expected_dificulty: int,
    population_size: int = 50,
    generations: int = 200,
    tournament_size: int = 3,
    crossover_rate: float = 0.9,
    mutation_rate: float = 0.01,
    elite_size: int = 2,
) -> NDArray[np.int32]:
    population: list[NDArray[np.int32]] = [
        generate_random_map() for _ in range(population_size)
    ]
    scores: list[float] = [evaluate(mapa, expected_dificulty) for mapa in population]

    best_index = int(np.argmin(scores))
    best = population[best_index].copy()
    best_score = scores[best_index]

    for _ in range(generations):
        elite = [int(i) for i in np.argsort(scores)[:elite_size]]
        offspring: list[NDArray[np.int32]] = [population[i].copy() for i in elite]
        offspring_scores: list[float] = [scores[i] for i in elite]

        while len(offspring) < population_size:
            first = tournament(population, scores, tournament_size)
            second = tournament(population, scores, tournament_size)

            if rng.random() < crossover_rate:
                crossover = (
                    two_point_crossover if rng.random() < 0.5 else block_crossover
                )
                child = crossover(first, second)
            else:
                child = first.copy()

            mutate(child, mutation_rate)
            _ = repair(child)

            offspring.append(child)
            offspring_scores.append(evaluate(child, expected_dificulty))

        population = offspring
        scores = offspring_scores

        best_index = int(np.argmin(scores))
        if scores[best_index] < best_score:
            best = population[best_index].copy()
            best_score = scores[best_index]
    return best
