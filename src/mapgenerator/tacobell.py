import time

from generator.filete import simulated_annealing
from generator.map_utils import (
    calculate_distribution_penalty,
    calculate_enemy_penalty,
    calculate_path_penalty,
    calculate_reward_penalty,
    fancy_print,
    generate_random_map,
    get_difficulty,
)
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
    print("p distribucion:", calculate_distribution_penalty(mapa))
    print("p path:", calculate_path_penalty(mapa, solution))
    if solution is not None:
        print("dificulty:", get_difficulty(mapa, solution))
    fancy_print(mapa)


def main():
    # normal()
    final = simulated_annealing(generate_random_map(), 40)
    print("penalizacion enemigos:", calculate_enemy_penalty(final))
    print("pene recompenza:", calculate_reward_penalty(final))
    solution = bfs(final)
    if solution is not None:
        print("dificulty:", get_difficulty(final, solution))
    fancy_print(final)
    # print(mapa.tolist())
