import numpy as np

class ParticleSwarm:
    def __init__(self):
        """
        Oswaldo es un hombre de muchas facetas, despues de su desliz con las generaciones de mapas, decidimos
        pasarlo a un campo mas tranquilo, ahora es el señor de las abejas
        """
        pass

    def traducir_particula_a_mapa(self, matriz_flotante):
        """
        Las abejas de oswaldo vienen de decimalandia, por lo cual, Oswaldo no puede entenderlas, el no habla
        taka taka, pero en su intento de hacer un buen trabajo, Oswaldo saca su diccionario decimal-entero 
        para hacer la traducccion, y que el evaluador no explote
        """
        filas = len(matriz_flotante)
        columnas = len(matriz_flotante[0])
        
        # En un gito inesperado de las cosas, oswaldo toma la pala, y 
        mapa_fisico = np.zeros((filas, columnas), dtype=int)
        
        for f in range(filas):
            for c in range(columnas):
                # 1. Oswaldo protege la entrada (0,0) y la salida (14,14) con su vida. Siempre son piso (0).
                if (f == 0 and c == 0) or (f == filas - 1 and c == columnas - 1):
                    mapa_fisico[f][c] = 0
                    continue
                
                valor = matriz_flotante[f][c]
                
                # 2. Diccionario de traducción de Oswaldo (basado estrictamente en la Tarea #25):
                
                # Menor a 0.70 -> Camino libre (0)
                if valor < 0.70:
                    mapa_fisico[f][c] = 0
                
                # De 0.70 a 0.84 -> Obstáculo (1)
                elif 0.70 <= valor <= 0.84:
                    mapa_fisico[f][c] = 1
                
                # De 0.85 a 0.94 -> Enemigo (2)
                elif 0.85 <= valor <= 0.94:
                    mapa_fisico[f][c] = 2
                
                # De 0.95 en adelante -> Recompensa (3)
                else:
                    mapa_fisico[f][c] = 3
                    
        return mapa_fisico

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