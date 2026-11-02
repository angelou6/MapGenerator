import argparse
import csv
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from generator.filete import simulated_annealing
from generator.map_utils import (
    CONSTRAINTS, DEFAULT_WEIGHTS, calculate_enemy_penalty, calculate_reward_penalty,
    calculate_distribution_penalty, calculate_path_penalty, fancy_print,
    generate_random_map, get_difficulty, generar_y_guardar_mapas
)
from generator.scenarios import SCENARIOS, get_scenario
from generator.genetic import Genetic
from path_search.breadth import bfs

# Importamos la función orquestadora del PSO que acabamos de crear
from generator.particulas import ejecutar_pso

def guardar_grafica_convergencia(history: list[tuple[float, float]]) -> None:
    iteraciones = range(len(history))
    fitness_actual = [actual for actual, mejor in history]
    mejor_fitness = [mejor for actual, mejor in history]
    plt.figure(figsize=(9, 5))
    plt.plot(iteraciones, fitness_actual, label="Fitness actual", alpha=0.5)
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
    parser = argparse.ArgumentParser(description="Orquestador del Proyecto Supremo de Mapas")
    # Agregamos "pso" a las opciones del algoritmo
    parser.add_argument("--algoritmo", choices=["recocido", "genetico", "pso"], default="recocido", help="Algoritmo a ejecutar")
    parser.add_argument("--pesos", type=float, nargs=5, default=DEFAULT_WEIGHTS, help="Pesos del fitness")
    parser.add_argument("--escenario", choices=SCENARIOS, help="Escenario (opcional)")
    parser.add_argument("--dificultad", type=int, help="Dificultad objetivo")
    args = parser.parse_args()

    if args.algoritmo == "genetico":
        print("\n[FASE 1] Forjando el universo base...")
        generar_y_guardar_mapas(cantidad=100)
        
        ruta_json = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Mapas", "poblacion_inicial.json")
        with open(ruta_json, "r", encoding="utf-8") as archivo:
            mapas_universales = json.load(archivo)
            
        print(f"[FASE 2] Universo cargado. {len(mapas_universales)} mapas listos.")
        ag = Genetic(poblacion_base=mapas_universales)
        mejor_mapa, historia = ag.entrenar(total_generaciones=20)
        print(f"\nEl Algoritmo Genético terminó con un Mejor Fitness de: {mejor_mapa['fitness']}")

    elif args.algoritmo == "recocido":
        config = get_scenario(args.escenario) if args.escenario else None
        rangos = config["rangos"] if config else CONSTRAINTS
        dificultad = int(args.dificultad) if args.dificultad is not None else (int(config["dificultad"]) if config and config["dificultad"] is not None else 40)
        
        history = []
        final = simulated_annealing(generate_random_map(args.escenario), dificultad, history=history, weights=tuple(args.pesos), rangos=rangos)
        
        with open("historial_recocido.csv", "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["iteracion", "fitness_actual", "mejor_fitness"])
            for iteration, (current, best) in enumerate(history):
                writer.writerow([iteration, current, best])
                
        guardar_grafica_convergencia(history)
        print("\nResultados Recocido Simulado guardados en CSV e imagen.")
        fancy_print(final)

    elif args.algoritmo == "pso":
        print("\n[FASE 1] Desplegando el Enjambre de Partículas...")
        # Ejecutamos el PSO con 30 partículas y 100 iteraciones (puedes ajustar estos valores)
        mejor_mapa, historial_fitness = ejecutar_pso(num_particulas=30, iteraciones=100)
        
        print(f"\n[FASE 2] El Enjambre terminó. Mejor Fitness global: {historial_fitness[-1]}")
        fancy_print(mejor_mapa)
        
        # Generar gráfica de convergencia específica para PSO
        plt.figure(figsize=(9, 5))
        plt.plot(range(len(historial_fitness)), historial_fitness, label="Mejor Fitness Global", linewidth=2, color="purple")
        plt.xlabel("Iteración")
        plt.ylabel("Fitness (menor es mejor)")
        plt.title("Convergencia de Optimización por Enjambre de Partículas (PSO)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("convergencia_pso.png", dpi=150)
        plt.close()
        print("Gráfica de convergencia del enjambre guardada como 'convergencia_pso.png'.")

if __name__ == "__main__":
    main()