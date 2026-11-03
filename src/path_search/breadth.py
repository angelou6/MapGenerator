from collections import deque
from collections.abc import Callable

import numpy as np
from numpy._typing import NDArray

from representation import BLOCKERS, MAP_SHAPE

Tile = tuple[int, int]


def find_path(
    start: Tile, goal: Tile, is_walkable: Callable[[int, int], bool]
) -> list[Tile] | None:
    """
    Busca el camino mas corto de start a goal usando breadth-first search.

    Args:
        start: Tile inicial como (columna, fila).
        goal: Tile objetivo como (columna, fila).
        is_walkable: Funcion que recibe (columna, fila) y dice si se puede pasar.

    Returns:
        list[Tile] or None: La ruta incluyendo start y goal, o None si no hay camino.
    """
    queue: deque[Tile] = deque([start])
    comes_from: dict[Tile, Tile | None] = {start: None}

    while queue:
        current = queue.popleft()

        if current == goal:
            path: list[Tile] = []
            tile: Tile | None = current
            while tile is not None:
                path.append(tile)
                tile = comes_from[tile]
            return path[::-1]

        col, row = current
        for neighbor in (
            (col, row - 1),
            (col, row + 1),
            (col - 1, row),
            (col + 1, row),
        ):
            if neighbor not in comes_from and is_walkable(*neighbor):
                comes_from[neighbor] = current
                queue.append(neighbor)

    return None


def bfs(mapa: NDArray[np.int32]) -> list[tuple[int, int]] | None:
    """
    Busca un camino posible dentro de un mapa usando breadth-first search.

    Args:
        NDArray[np.int32]: El mapa generado como array 15x15 de Numpy.

    Returns:
        list[tuple[int, int]] or None: Devuelve la ruta encontrada,
        o devuelve None si no se encontró ninguna ruta.
    """
    shape = MAP_SHAPE[0]
    grid = mapa.tolist()

    def is_walkable(col: int, row: int) -> bool:
        return 0 <= row < shape and 0 <= col < shape and grid[row][col] not in BLOCKERS

    path = find_path((0, 0), (shape - 1, shape - 1), is_walkable)
    if path is None:
        return None
    return [(row, col) for col, row in path]
