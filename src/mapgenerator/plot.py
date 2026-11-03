from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mapgenerator.main import ALGORITHMS, DEFAULT_DIFICULTY

RUNS = 10
IMAGES = Path("images")


def run(name: str) -> dict[str, list[float]]:
    history: dict[str, list[float]] = {}
    _ = ALGORITHMS[name](DEFAULT_DIFICULTY, history=history)
    return history


def plot(name: str, histories: list[dict[str, list[float]]]) -> None:
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
    _ = fig.suptitle(f"{name} ({RUNS} ejecuciones)")  # pyright: ignore[reportUnknownMemberType]
    fig.tight_layout()
    fig.savefig(IMAGES / f"{name}.png")  # pyright: ignore[reportUnknownMemberType]
    plt.close(fig)


def main() -> None:
    IMAGES.mkdir(exist_ok=True)
    with ProcessPoolExecutor(mp_context=get_context("spawn")) as executor:
        futures = {
            name: [executor.submit(run, name) for _ in range(RUNS)]
            for name in ALGORITHMS
        }
        for name, runs in futures.items():
            plot(name, [future.result() for future in runs])
            print(f"Guardado {IMAGES / f'{name}.png'}")
