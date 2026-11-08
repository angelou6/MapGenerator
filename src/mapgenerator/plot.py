# pyright: reportUnknownMemberType=false
import csv
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import TypedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from numpy.typing import NDArray

from generator.map_utils import (
    FANCY_ELEMENTOS,
    calculate_average_diversity,
    calculate_fitness,
    get_dificulty,
    is_valid,
)
from generator.scenarios import SCENARIOS, get_scenario
from mapgenerator.main import ALGORITHMS, DEFAULT_DIFICULTY
from path_search.breadth import bfs
from representation import Elemento

RUNS = 10
IMAGES = Path("images")
CELL_COLORS = ["#f4f3ef", "#52514e", "#e34948", "#eda100", "#2a78d6", "#008300"]


class RunResult(TypedDict):
    mapa: NDArray[np.int32]
    fitness: float
    valido: bool
    dificultad: float | None
    tiempo: float
    convergencia: list[float]


def run_escenario(algoritmo: str, escenario: str) -> RunResult:
    config = get_scenario(escenario)
    objetivo = config["dificultad"]
    if objetivo is None:
        objetivo = DEFAULT_DIFICULTY
    history: dict[str, list[float]] = {}
    start = time.perf_counter()
    mapa = ALGORITHMS[algoritmo](objetivo, escenario=escenario, history=history)
    tiempo = time.perf_counter() - start
    solution = bfs(mapa)
    return {
        "mapa": mapa,
        "fitness": calculate_fitness(
            mapa,
            solution,
            objetivo,
            camino_largo=config["camino_largo"],
        ),
        "valido": is_valid(mapa, config["rangos"]),
        "dificultad": get_dificulty(mapa, solution) if solution else None,
        "tiempo": tiempo,
        "convergencia": history["mejor fitness"],
    }


def run(name: str) -> tuple[dict[str, list[float]], float]:
    history: dict[str, list[float]] = {}
    start = time.perf_counter()
    _ = ALGORITHMS[name](DEFAULT_DIFICULTY, history=history)
    return history, time.perf_counter() - start


def plot(name: str, histories: list[dict[str, list[float]]], tiempos: list[float]) -> None:
    metrics = list(histories[0])
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4))
    for ax, metric in zip(axes, metrics):
        values = np.array([history[metric] for history in histories])
        for row in values:
            _ = ax.plot(row, color="#2a78d6", linewidth=1, alpha=0.3)
        _ = ax.plot(values.mean(axis=0), color="#1f2937", linewidth=2, label="promedio")
        _ = ax.set_title(metric)
        _ = ax.set_xlabel("iteración")
        ax.grid(alpha=0.3)
        _ = ax.legend()
    _ = fig.suptitle(
        f"{name} ({RUNS} ejecuciones, tiempo promedio {np.mean(tiempos):.2f} s)"
    )
    fig.tight_layout()
    fig.savefig(IMAGES / f"{name}.png")
    plt.close(fig)


def draw_map(ax: Axes, mapa: NDArray[np.int32], title: str) -> None:
    _ = ax.imshow(mapa, cmap=ListedColormap(CELL_COLORS), vmin=0, vmax=5)
    for (r, c), v in np.ndenumerate(mapa):
        if v != Elemento.TRANSITABLE:
            _ = ax.text(
                c,
                r,
                FANCY_ELEMENTOS[Elemento(v)],
                ha="center",
                va="center",
                fontsize=7,
                color="black" if v == Elemento.RECOMPENZA else "white",
            )
    solution = bfs(mapa)
    if solution:
        rows, cols = zip(*solution)
        _ = ax.plot(cols, rows, color="#6250d6", linewidth=2)
    _ = ax.set_xticks([])
    _ = ax.set_yticks([])
    _ = ax.set_title(title, fontsize=9)


def main() -> None:
    IMAGES.mkdir(exist_ok=True)
    with ProcessPoolExecutor(mp_context=get_context("spawn")) as executor:
        futures = {
            (e, a): [executor.submit(run_escenario, a, e) for _ in range(RUNS)]
            for e in SCENARIOS
            for a in ALGORITHMS
        }
        historias = {
            name: [executor.submit(run, name) for _ in range(RUNS)]
            for name in ALGORITHMS
        }
        for name, runs in historias.items():
            histories, tiempos = zip(*(future.result() for future in runs))
            plot(name, list(histories), list(tiempos))
            print(f"Guardado {IMAGES / f'{name}.png'}")
        results = {k: [f.result() for f in fs] for k, fs in futures.items()}

    filas: list[dict[str, str | float | None]] = []
    for (escenario, algoritmo), rs in results.items():
        fitness = np.array([r["fitness"] for r in rs])
        dificultades = [r["dificultad"] for r in rs if r["dificultad"] is not None]
        filas.append(
            {
                "escenario": escenario,
                "algoritmo": algoritmo,
                "mejor": fitness.min(),
                "peor": fitness.max(),
                "promedio": fitness.mean(),
                "desviacion": fitness.std(ddof=1),
                "validos_%": 100 * np.mean([r["valido"] for r in rs]),
                "dificultad": np.mean(dificultades) if dificultades else None,
                "tiempo_s": np.mean([r["tiempo"] for r in rs]),
                "diversidad_hamming": calculate_average_diversity(
                    [r["mapa"] for r in rs]
                ),
            }
        )
    with (IMAGES / "tabla.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(filas[0]))
        writer.writeheader()
        writer.writerows(filas)

    for escenario in SCENARIOS:
        fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
        for ax, algoritmo in zip(axes[1:], ALGORITHMS):
            rs = results[(escenario, algoritmo)]
            curva = np.mean([r["convergencia"] for r in rs], axis=0)
            axes[0].plot(np.linspace(0, 1, len(curva)), curva, label=algoritmo)
            best = min(rs, key=lambda r: r["fitness"])
            draw_map(ax, best["mapa"], f"{algoritmo}: fitness {best['fitness']:.1f}")
        axes[0].set_yscale("symlog")
        axes[0].set_xlabel("progreso (fracción de iteraciones)")
        axes[0].set_ylabel("mejor fitness promedio")
        axes[0].set_title(f"Convergencia - {escenario}")
        axes[0].legend()
        fig.tight_layout()
        fig.savefig(IMAGES / f"{escenario}.png", dpi=110)
        plt.close(fig)

    print(f"Resultados en {IMAGES}/")
