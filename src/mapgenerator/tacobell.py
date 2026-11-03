import time

from generator.filete import simulated_annealing
from generator.genetic import genetic_algorithm
from generator.map_utils import (
    calculate_average_diversity,
    calculate_distrubution_penalty,
    calculate_enemy_penalty,
    calculate_path_penalty,
    calculate_reward_penalty,
    fancy_print,
    generate_random_map,
    get_dificulty,
)
from generator.particulas import particle_swarm
from path_search.breadth import bfs


def check_recompenza():
    re = 0
    mapa = None
    while re == 0:
        mapa = generate_random_map()
        re = calculate_reward_penalty(mapa)

    if mapa is not None:
        print("penalizacion enemigos:", calculate_enemy_penalty(mapa))
        print("pene recompenza:", calculate_reward_penalty(mapa))
        fancy_print(mapa)


def execution_time():
    mapa = generate_random_map()
    start_time = time.time()
    nt = calculate_reward_penalty(mapa)
    print(time.time() - start_time, "seconds")
    print(nt)
    fancy_print(mapa)


def normal():
    mapa = generate_random_map()
    solution = bfs(mapa)
    print("penalizacion enemigos:", calculate_enemy_penalty(mapa))
    print("pene recompenza:", calculate_reward_penalty(mapa))
    print("p distribucion:", calculate_distrubution_penalty(mapa))
    print("p path:", calculate_path_penalty(solution))
    if solution is not None:
        print("dificulty:", get_dificulty(mapa, solution))
    fancy_print(mapa)


def diversidad(ejecuciones: int = 5, dificultad: int = 40):
    algoritmos = {
        "aleatorio": lambda: generate_random_map(),
        "recocido": lambda: simulated_annealing(dificultad, steps=2000),
        "genetico": lambda: genetic_algorithm(dificultad, generations=30),
        "particulas": lambda: particle_swarm(dificultad, iterations=30),
    }
    for nombre, generar in algoritmos.items():
        mapas = [generar() for _ in range(ejecuciones)]
        print(f"diversidad {nombre}:", calculate_average_diversity(mapas))


def main():
    # diversidad()
    # normal()
    final = simulated_annealing(40)
    print("penalizacion enemigos:", calculate_enemy_penalty(final))
    print("pene recompenza:", calculate_reward_penalty(final))
    solution = bfs(final)
    if solution is not None:
        print("dificulty:", get_dificulty(final, solution))
    fancy_print(final)
    # print(mapa.tolist())
