from collections import deque

import numpy as np
from numpy._typing import NDArray

from representation import BLOCKERS, MAP_SHAPE


def bfs(mapa: NDArray[np.int32]) -> list[tuple[int, int]] | None:
    """
    Busca un camino posible dentro de un mapa usando breadth-first search.

    Args:
        NDArray[np.int32]: El mapa generado como array 15x15 de Numpy.

    Returns:
        list[tuple[int, int]] or None: Devuelve la ruta encontrada, o devuelve None si no se encontró ninguna ruta.
    """
    start = (0, 0)
    goal = (14, 14)
    shape = MAP_SHAPE[0]
    grid = mapa.tolist()

    queue: deque[tuple[int, int]] = deque([start])
    comes_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}

    while queue:
        current = queue.popleft()
        row, col = current

        if current == goal:
            path: list[tuple[int, int]] = []
            while current is not None:
                path.append(current)
                current = comes_from[current]
            return path[::-1]

        for neighbor in (
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1),
        ):
            nrow, ncol = neighbor
            if (
                0 <= nrow < shape
                and 0 <= ncol < shape
                and grid[nrow][ncol] not in BLOCKERS
                and neighbor not in comes_from
            ):
                comes_from[neighbor] = current
                queue.append(neighbor)
    return None
