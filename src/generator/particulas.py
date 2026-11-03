"""
Optimizacion por Enjambre de Particulas (PSO).

Cada celda es un numero continuo en [0,1]; para evaluar se discretiza con los intervalos
del documento (0-0.69 libre, 0.70-0.84 obstaculo, 0.85-0.94 enemigo, 0.95-1 recompensa)
y se repara para respetar las cantidades permitidas. Se minimiza f(X).
"""
import numpy as np
from numpy.typing import NDArray

from generator.map_utils import CONSTRAINTS
from representation import MAP_SHAPE, Elemento

rng = np.random.default_rng()

# Intervalos [a, b) de cada elemento (seccion 10.2 del documento)
INTERVALOS = {
    Elemento.TRANSITABLE: (0.0, 0.70),
    Elemento.OBSTACULO: (0.70, 0.85),
    Elemento.ENEMIGO: (0.85, 0.95),
    Elemento.RECOMPENZA: (0.95, 1.0),
}


def generate_random_particle() -> NDArray[np.float64]:
    """Crea una particula de 15x15 con valores entre 0 y 1 (inicio y meta en 0)."""
    particula = rng.random(MAP_SHAPE)
    particula[0, 0] = 0.0
    particula[-1, -1] = 0.0
    return particula


def mapa_a_particula(mapa: NDArray[np.int32]) -> NDArray[np.float64]:
    """
    Convierte un mapa en una particula cuya discretizacion devuelve exactamente ese mapa
    (cada celda recibe un valor al azar dentro del intervalo de su elemento).
    """
    particula = np.zeros(MAP_SHAPE)
    for elemento, (a, b) in INTERVALOS.items():
        mascara = mapa == elemento
        particula[mascara] = rng.uniform(a, b - 1e-9, int(mascara.sum()))
    particula[0, 0] = 0.0
    particula[-1, -1] = 0.0
    return particula


def particle_to_map(particula: NDArray[np.float64],
                    rangos: dict[Elemento, tuple[int, int]] = CONSTRAINTS) -> NDArray[np.int32]:
    """
    Discretiza una particula y la repara.

    Reparacion guiada por los valores continuos: se cuentan las celdas de cada intervalo,
    se recorta cada cantidad a su rango [min, max] y las celdas se reparten por orden de
    valor (las mas altas son recompensas, luego enemigos, luego obstaculos). Si las
    cantidades ya estaban en rango el resultado es identico a la discretizacion simple.
    """
    if particula.shape != MAP_SHAPE:
        raise ValueError("La particula debe tener un tamaño de 15x15")
    if not np.all(np.isfinite(particula) & (particula >= 0) & (particula <= 1)):
        raise ValueError("Los valores de la particula debe estar entre 0 y 1")

    movible = np.ones(MAP_SHAPE, dtype=bool)
    movible[0, 0] = movible[-1, -1] = False
    valores = particula[movible]

    def recortar(elemento, cantidad):
        lo, hi = rangos.get(elemento, (0, valores.size))
        return int(min(max(cantidad, lo), hi))

    n_r = recortar(Elemento.RECOMPENZA, int(np.count_nonzero(valores >= 0.95)))
    n_e = recortar(Elemento.ENEMIGO, int(np.count_nonzero((valores >= 0.85) & (valores < 0.95))))
    n_o = recortar(Elemento.OBSTACULO, int(np.count_nonzero((valores >= 0.70) & (valores < 0.85))))

    # desempate aleatorio minimo (al recortar en [0,1] muchas celdas valen exactamente 0 o 1)
    orden = np.argsort(-(valores + rng.random(valores.size) * 1e-9))
    celdas = np.zeros(valores.size, dtype=np.int32)
    celdas[orden[:n_r]] = Elemento.RECOMPENZA
    celdas[orden[n_r:n_r + n_e]] = Elemento.ENEMIGO
    celdas[orden[n_r + n_e:n_r + n_e + n_o]] = Elemento.OBSTACULO

    mapa = np.zeros(MAP_SHAPE, dtype=np.int32)
    mapa[movible] = celdas
    mapa[0, 0] = Elemento.INICIO
    mapa[-1, -1] = Elemento.META
    return mapa


def update_particle_velocity_position(
    posicion_actual, velocidad_actual, mejor_posicion_individual, mejor_posicion_global,
    w: float = 0.6, c1: float = 1.5, c2: float = 1.5, vmax: float = 0.5,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    v(t+1) = w v(t) + c1 r1 (pbest - x) + c2 r2 (gbest - x);  x(t+1) = x(t) + v(t+1)
    (formula del documento). Se limita |v| <= vmax y x a [0,1]; inicio/meta quedan en 0.
    """
    r1 = rng.random(MAP_SHAPE)
    r2 = rng.random(MAP_SHAPE)
    velocidad = (w * velocidad_actual
                 + c1 * r1 * (mejor_posicion_individual - posicion_actual)
                 + c2 * r2 * (mejor_posicion_global - posicion_actual))
    velocidad = np.clip(velocidad, -vmax, vmax)
    posicion = np.clip(posicion_actual + velocidad, 0.0, 1.0)
    posicion[0, 0] = 0.0
    posicion[-1, -1] = 0.0
    return posicion, velocidad


def ejecutar_pso(evaluador, rangos, mapas_iniciales, iteraciones=100, progreso=None):
    """
    Ejecuta el enjambre completo; hay una particula por cada mapa inicial.

    Returns:
        (mapas_finales, historial): el mejor mapa historico (pbest) de cada particula, en
        el mismo orden que mapas_iniciales, y una lista de dicts
        {iteracion, mejor, media, peor} sobre los pbest (iteracion 0 = estado inicial).
    """
    n = len(mapas_iniciales)
    posiciones = [mapa_a_particula(np.asarray(m)) for m in mapas_iniciales]
    velocidades = [np.zeros(MAP_SHAPE) for _ in range(n)]

    pbest_pos = [p.copy() for p in posiciones]
    pbest_map = [particle_to_map(p, rangos) for p in posiciones]
    pbest_fit = [evaluador.fitness(m) for m in pbest_map]
    g = int(np.argmin(pbest_fit))
    gbest_pos = pbest_pos[g].copy()

    historial = []

    def registrar(it):
        historial.append({"iteracion": it, "mejor": float(min(pbest_fit)),
                          "media": float(np.mean(pbest_fit)), "peor": float(max(pbest_fit))})

    registrar(0)
    for it in range(1, iteraciones + 1):
        for i in range(n):
            posiciones[i], velocidades[i] = update_particle_velocity_position(
                posiciones[i], velocidades[i], pbest_pos[i], gbest_pos)
            mapa = particle_to_map(posiciones[i], rangos)
            fit = evaluador.fitness(mapa)
            if fit < pbest_fit[i]:
                pbest_fit[i], pbest_pos[i], pbest_map[i] = fit, posiciones[i].copy(), mapa
        g = int(np.argmin(pbest_fit))
        gbest_pos = pbest_pos[g].copy()
        registrar(it)
        if progreso:
            progreso(it, iteraciones)
    return pbest_map, historial
