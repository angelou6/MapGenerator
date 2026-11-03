import argparse
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from game.game import Game
from generator.map_utils import generate_random_map
from generator.scenarios import SCENARIOS, get_scenario
from mapgenerator.main import ALGORITHMS, DEFAULT_DIFICULTY
from representation import MAP_SHAPE, Elemento


def load_map(path: Path) -> NDArray[np.int32]:
    """
    Carga un mapa guardado con numpy.save y valida su forma y contenido.

    Args:
        path: Ruta al archivo .npy.

    Returns:
        NDArray[np.int32]: El mapa cargado.

    Raises:
        ValueError: Si el archivo no contiene un mapa válido.
    """
    mapa = np.load(path, allow_pickle=False)
    if mapa.shape != MAP_SHAPE:
        raise ValueError(f"El mapa debe tener forma {MAP_SHAPE}, tiene {mapa.shape}")
    if not np.issubdtype(mapa.dtype, np.integer):
        raise ValueError(f"El mapa debe ser de enteros, es {mapa.dtype}")
    validos = [e.value for e in Elemento]
    if not np.isin(mapa, validos).all():
        raise ValueError(f"El mapa solo puede contener los valores {validos}")
    return mapa.astype(np.int32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Juego con mapas generados")
    _ = parser.add_argument(
        "--algoritmo",
        choices=["aleatorio", *ALGORITHMS],
        default="aleatorio",
        help="Algoritmo para generar el mapa (por defecto: %(default)s)",
    )
    _ = parser.add_argument(
        "--dificultad",
        type=int,
        help=f"Dificultad objetivo (por defecto: la del escenario o {DEFAULT_DIFICULTY})",
    )
    _ = parser.add_argument("--escenario", choices=SCENARIOS, help="Escenario")
    _ = parser.add_argument(
        "--mapa",
        type=Path,
        help="Archivo .npy (guardado con numpy.save) con el mapa a jugar. "
        + "Ignora las demás opciones",
    )

    args = parser.parse_args()

    if args.mapa is not None:
        try:
            mapa = load_map(args.mapa)
        except (OSError, ValueError) as e:
            parser.error(f"No se pudo cargar el mapa '{args.mapa}': {e}")
    elif args.algoritmo == "aleatorio":
        mapa = generate_random_map(args.escenario)
    else:
        dificultad: int | None = args.dificultad
        if dificultad is None and args.escenario is not None:
            dificultad = get_scenario(args.escenario)["dificultad"]
        mapa = ALGORITHMS[args.algoritmo](
            dificultad if dificultad is not None else DEFAULT_DIFICULTY,
            escenario=args.escenario,
        )

    Game(mapa.tolist()).run()
