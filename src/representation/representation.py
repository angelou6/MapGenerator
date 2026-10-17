from enum import IntEnum


class Elemento(IntEnum):
    TRANSITABLE = 0
    OBSTACULO = 1
    ENEMIGO = 2
    RECOMPENZA = 3
    INICIO = 4
    META = 5


BLOCKERS = (Elemento.ENEMIGO, Elemento.OBSTACULO)
MAP_SHAPE = (15, 15)
