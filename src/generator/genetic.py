"""
Generador de mapas mediante Algoritmo Genético.

Evoluciona una población de mapas representados como matrices de 15x15 utilizando
selección por torneo, cruces cromosómicos (por dos puntos o por bloques de 5x5),
mutaciones puntuales y reparación de restricciones con elitismo.
"""

import numpy as np
from numpy.typing import NDArray

from generator.map_utils import (
    MOVABLE_MASK,
    evaluate,
    generate_random_map,
    repair,
)
from generator.scenarios import get_scenario
from representation import MAP_SHAPE, Elemento

rng = np.random.default_rng()

BLOCK = 5


def tournament(
    population: list[NDArray[np.int32]], scores: list[float], size: int
) -> NDArray[np.int32]:
    """
    Selecciona al mejor individuo de un subconjunto de participantes elegidos al azar.

    Args:
        population: Lista con los mapas de la población actual.
        scores: Lista con los valores de fitness correspondientes a cada mapa.
        size: Cantidad de individuos participantes en el torneo.

    Returns:
        NDArray[np.int32]: El mapa del individuo ganador del torneo.
    """
    contenders = rng.choice(len(population), size=size, replace=False)
    winner = int(contenders[np.argmin(np.asarray(scores)[contenders])])
    return population[winner]


def two_point_crossover(
    first: NDArray[np.int32], second: NDArray[np.int32]
) -> NDArray[np.int32]:
    """
    Aplica cruce de dos puntos sobre los cromosomas aplanados de dos mapas progenitores.

    Args:
        first: Mapa del primer progenitor.
        second: Mapa del segundo progenitor.

    Returns:
        NDArray[np.int32]: Nuevo mapa generado a partir del cruce.
    """
    a, b = sorted(rng.choice(first.size, size=2, replace=False))
    child = first.flatten()
    child[a:b] = second.flatten()[a:b]
    return child.reshape(MAP_SHAPE)


def block_crossover(
    first: NDArray[np.int32], second: NDArray[np.int32]
) -> NDArray[np.int32]:
    """
    Aplica cruce por regiones donde cada bloque de 5x5 se hereda de uno de los padres.

    Args:
        first: Mapa del primer progenitor.
        second: Mapa del segundo progenitor.

    Returns:
        NDArray[np.int32]: Nuevo mapa descendiente con bloques combinados.
    """
    child = first.copy()
    rows, cols = MAP_SHAPE
    for fila in range(0, rows, BLOCK):
        for columna in range(0, cols, BLOCK):
            if rng.random() < 0.5:
                region = (slice(fila, fila + BLOCK), slice(columna, columna + BLOCK))
                child[region] = second[region]
    return child


def mutate(candidate: NDArray[np.int32], mutation_rate: float) -> None:
    """
    Muta aleatoriamente celdas movibles asignándoles un nuevo elemento con cierta probabilidad.

    Args:
        candidate: Array 2D de Numpy que representa el mapa. Se modifica en el lugar.
        mutation_rate: Probabilidad de mutación por cada celda del mapa.
    """
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
    escenario: str | None = None,
) -> NDArray[np.int32]:
    """
    Ejecuta el ciclo evolutivo del algoritmo genético para encontrar un mapa óptimo.

    Args:
        expected_dificulty: Dificultad objetivo esperada para el mapa generado.
        population_size: Cantidad de mapas en cada generación.
        generations: Número de generaciones a iterar.
        tournament_size: Número de individuos participantes en la selección por torneo.
        crossover_rate: Probabilidad de aplicar cruce entre dos padres seleccionados.
        mutation_rate: Tasa de mutación aleatoria por celda.
        elite_size: Cantidad de mejores individuos conservados intactos por generación.
        escenario: Nombre opcional del escenario para ajustar las restricciones y dificultad.

    Returns:
        NDArray[np.int32]: El mejor mapa encontrado al concluir todas las generaciones.
    """
    rangos = get_scenario(escenario)["rangos"] if escenario is not None else None
    population: list[NDArray[np.int32]] = [
        generate_random_map(escenario) for _ in range(population_size)
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
            _ = repair(child, rangos=rangos)

            offspring.append(child)
            offspring_scores.append(evaluate(child, expected_dificulty))

        population = offspring
        scores = offspring_scores

        best_index = int(np.argmin(scores))
        if scores[best_index] < best_score:
            best = population[best_index].copy()
            best_score = scores[best_index]
    return best
