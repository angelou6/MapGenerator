import argparse
import csv
import time
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from generator.filete import simulated_annealing
from generator.map_utils import (
    CONSTRAINTS,
    DEFAULT_WEIGHTS,
    calculate_distrubution_penalty,
    calculate_enemy_penalty,
    calculate_path_penalty,
    calculate_reward_penalty,
    fancy_print,
    generate_random_map,
    get_dificulty,
)
from generator.scenarios import SCENARIOS, get_scenario 
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
    print("p path:", calculate_path_penalty(mapa, solution))
    if solution is not None:
        print("dificulty:", get_dificulty(mapa, solution))
    fancy_print(mapa)


def guardar_grafica_convergencia(history: list[tuple[float, float]]) -> None:
    #dibuja el fitness actual y el mejor de cada iteracion usando el historial guardado :3
    iteraciones = range(len(history))
    fitness_actual = [actual for actual, mejor in history]
    mejor_fitness = [mejor for actual, mejor in history]

    plt.figure(figsize=(9, 5))
    plt.plot(iteraciones, fitness_actual, label="Fitness actual", alpha = 0.5)
    plt.plot(iteraciones, mejor_fitness, label="Mejor fitness", linewidth=2)
    plt.xlabel("iteracion")
    plt.ylabel("Fitness (menor es mejor)")
    plt.title("Convergencia de Recocido Simulado")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("convergencia_recocido.png", dpi=150)
    plt.close()


def main():
    #perrmite cambiar los pesos desde la terminal y si no se usa, pues usa los default
    parser = argparse.ArgumentParser(description="Pruebas de Recocido Simulado")
    parser.add_argument(
        "--pesos", type=float, nargs=5, default=DEFAULT_WEIGHTS,
        metavar=("CAMINO", "DIFICULTAD", "ENEMIGOS", "RECOMPENSAS", "DISTRIBUCION"),
        help="Pesos del fitness (Por defecto: 50 10 5 5 2)",
    )
    #permite escoger el escenario y si hace falta, indicar su dificultad objetivo
    parser.add_argument(
        "--escenario", choices=SCENARIOS,
        help="Escenario paraa Recocido Simulado (no opcion: rangos normales)"
    )
    parser.add_argument(
        "--dificultad", type=int,
        help="Dificultad objetivo (obligatorio en escenarios de padres primos)",
    )
    args = parser.parse_args()
    if any(peso < 0 for peso in args.pesos):
        parser.error("Los pesos no pueden ser negativos")
    weights = tuple(args.pesos)
    print("Pesos del fitness:", weights)

    #obtienelos rangos de un solo lugar.
    #los escenarios especiales no tienen dificltad numérica definida en el documento.
    config = get_scenario(args.escenario) if args.escenario else None
    rangos = config["rangos"] if config else CONSTRAINTS
    if args.dificultad is not None:
        dificultad = int(args.dificultad)
    elif config is None:
        dificultad = 40
    elif config["dificultad"] is not None:
        dificultad = int(config["dificultad"])
    else:
        parser.error("Este escenario necesita --dificultad (valor objetivo)")
    print("Escenario:", args.escenario or "normmal")
    print("Dificultad objetivo:", dificultad)

    #normal()
    #pide a recorrido simulado el historial del fitness
    history = []
    #genera el mapa y lo optimiza xon,los mismos rengos del escenraio mantendiendo lso pesos y el historial de los anteriores
    final = simulated_annealing(
        generate_random_map(args.escenario), dificultad,
        history=history, weights=weights, rangos=rangos,
    )

    #guardar las calificaciones en un csv para hacer graficas despues sin instalar librerias nuevas
    with open("historial_recocido.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["iteracion", "fitnes_actuañ", "mejor_fitness"])
        for iteration, (current, best) in enumerate(history):
            writer.writerow([iteration, current, best])
    print("Historial guardado en historial_recocido.csv")
    #Creacion de la grafica con las calificaciones del csv
    guardar_grafica_convergencia(history)
    print("Grafica guardada en convergencia_recocido.png")
    print("penalizacion enemigos:", calculate_enemy_penalty(final))
    print("pene recompenza:", calculate_reward_penalty(final))
    solution = bfs(final)
    if solution is not None:
        print("dificulty:", get_dificulty(final, solution))
    fancy_print(final)
    #print(mapa.tolist())