"""Representacion numerica de las partículas para PSO"""

import numpy as np
from numpy._typing import NDArray

from representation import MAP_SHAPE, Elemento

rng = np.random.default_rng()

def generate_random_particle() -> NDArray[np.float64]:
    """Crea una partícula de 15x15 con valores entre 0 y 1"""
    #cada celda recibe un valor decimal para utilizarlo luego en el PSO
    particula = rng.random(MAP_SHAPE)

    #reserva las esquinas de inicio y fin
    particula[0, 0] = 0.0
    particula[-1, -1] = 0.0

    return particula

def particle_to_map(particula: NDArray[np.float64]) -> NDArray[np.int32]:
    """Convierte los valores de una particula en elementos del mapa"""
    #comprueba que la particula sea de 15x15 y que los valores esten entre tu y yo *guiño* (0 y 1)
    if particula.shape != MAP_SHAPE:
        raise ValueError("La particula debe tener un tamaño de 15x15")
    if not np.all(np.isfinite(particula) & (particula >= 0) & (particula <= 1)):
        raise ValueError("Los valores de la particula debe estarr entre 0 y 1")

    #conviertelos numeros en elementos usando los limites del codumento y los valores menores 
    #a 0.70 quedan libres
    mapa = np.zeros(MAP_SHAPE, dtype=np.int32)
    mapa[particula >= 0.70] = Elemento.OBSTACULO
    mapa[particula >= 0.85] = Elemento.ENEMIGO
    mapa[particula >= 0.95] = Elemento.RECOMPENZA

    #fija inicio y meta sin importar el valor decimal
    mapa[0, 0] = Elemento.INICIO
    mapa[-1, -1] = Elemento.META
    return mapa