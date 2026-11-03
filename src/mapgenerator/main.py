import argparse
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from generator.filete import simulated_annealing
from generator.genetic import genetic_algorithm
from generator.map_utils import (
    DEFAULT_WEIGHTS,
    WEIGHT_PRESETS,
    Weights,
    calculate_distrubution_penalty,
    calculate_enemy_penalty,
    calculate_fitness,
    calculate_path_penalty,
    calculate_reward_penalty,
    fancy_print,
    get_dificulty,
)
from generator.particulas import particle_swarm
from generator.scenarios import SCENARIOS, get_scenario
from path_search.breadth import bfs

Algorithm = Callable[..., NDArray[np.int32]]

ALGORITHMS: dict[str, Algorithm] = {
    "recocido": simulated_annealing,
    "genetico": genetic_algorithm,
    "pso": particle_swarm,
}
DEFAULT_DIFICULTY = 40


def data_print(
    callback: Algorithm,
    dificulty: int,
    escenario: str | None = None,
    weights: Weights = DEFAULT_WEIGHTS,
) -> None:
    camino_largo = get_scenario(escenario)["camino_largo"]

    final = callback(dificulty, escenario=escenario, weights=weights)
    print("Penalización enemigos:", calculate_enemy_penalty(final))
    print("Penalización recompensa:", calculate_reward_penalty(final))
    print("Penalización distribución:", calculate_distrubution_penalty(final))
    solution = bfs(final)
    print("Penalización camino:", calculate_path_penalty(solution, camino_largo))
    if solution is not None:
        print("Dificultad:", get_dificulty(final, solution))
        print("Longitud camino:", len(solution))
    else:
        print("Dificultad: Sin camino a la meta")
    print(
        "Fitness total:",
        calculate_fitness(final, solution, dificulty, weights, camino_largo),
    )
    fancy_print(final)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Orquestador del Proyecto Supremo de Mapas"
    )
    _ = parser.add_argument(
        "--algoritmo",
        choices=ALGORITHMS,
        default="recocido",
        help="Algoritmo a ejecutar (por defecto: %(default)s)",
    )
    _ = parser.add_argument(
        "--dificultad",
        type=int,
        help=f"Dificultad objetivo (por defecto: la del escenario o {DEFAULT_DIFICULTY})",
    )
    _ = parser.add_argument(
        "--pesos",
        type=float,
        nargs=5,
        default=DEFAULT_WEIGHTS,
        metavar=("CAMINO", "DIFICULTAD", "ENEMIGOS", "RECOMPENSAS", "DISTRIBUCION"),
        help="Pesos del fitness (por defecto: %(default)s)",
    )
    _ = parser.add_argument(
        "--prioridad",
        choices=WEIGHT_PRESETS,
        help="Usa los pesos de un experimento de la sección 15 (ignora --pesos)",
    )
    _ = parser.add_argument("--escenario", choices=SCENARIOS, help="Escenario")

    args = parser.parse_args()

    if any(p < 0 for p in args.pesos):
        parser.error("Los pesos del fitness no pueden ser negativos")
    pesos = WEIGHT_PRESETS[args.prioridad] if args.prioridad else args.pesos
    camino, dificultad_w, enemigos, recompensas, distribucion = pesos
    weights: Weights = (camino, dificultad_w, enemigos, recompensas, distribucion)

    dificultad: int | None = args.dificultad
    if dificultad is None and args.escenario is not None:
        dificultad = get_scenario(args.escenario)["dificultad"]

    data_print(
        ALGORITHMS[args.algoritmo],
        dificulty=dificultad if dificultad is not None else DEFAULT_DIFICULTY,
        escenario=args.escenario,
        weights=weights,
    )


if __name__ == "__main__":
    main()
