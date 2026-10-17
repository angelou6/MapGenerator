from collections import Counter

import numpy as np
from numpy.typing import NDArray

from representation import BLOCKERS, MAP_SHAPE, Elemento


def get_ajacent_blockers(
    mapa: NDArray[np.int32], path: list[tuple[int, int]]
) -> Counter[Elemento]:
    shape = MAP_SHAPE[0]
    grid = mapa.tolist()
    registered: set[tuple[int, int]] = set()
    count: Counter[Elemento] = Counter()

    for row, col in path:
        for neighbor in (
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1),
        ):
            nrow, ncol = neighbor
            if 0 <= nrow < shape and 0 <= ncol < shape and neighbor not in registered:
                current = grid[nrow][ncol]
                if current in BLOCKERS:
                    registered.add(neighbor)
                    count[grid[nrow][ncol]] += 1
    return count
