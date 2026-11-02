"""Representacion numerica de las partículas para PSO"""
from generator.map_utils import calculate_fitness

import numpy as np
from numpy._typing import NDArray

from generator.map_utils import CONSTRAINTS, reparar_cantidad_elemento
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
        raise ValueError("Los valores de la particula debe estar entre 0 y 1")

    #conviertelos numeros en elementos usando los limites del documento y los valores menores 
    #a 0.70 quedan libres
    mapa = np.zeros(MAP_SHAPE, dtype=np.int32)
    mapa[particula >= 0.70] = Elemento.OBSTACULO
    mapa[particula >= 0.85] = Elemento.ENEMIGO
    mapa[particula >= 0.95] = Elemento.RECOMPENZA

    #fija inicio y meta sin importar el valor decimal (por seguridad inicial)
    mapa[0, 0] = Elemento.INICIO
    mapa[-1, -1] = Elemento.META

    # APLICACIÓN DE LA REPARACIÓN (TAREA #27)
    # Oswaldo entra a escena: tomamos los rangos de constraints para reparar excesos o carencias.
    
    # 1. Reparar Obstáculos (usualmente entre 20 y 35)
    min_obs, max_obs = CONSTRAINTS[Elemento.OBSTACULO]
    mapa = reparar_cantidad_elemento(mapa, Elemento.OBSTACULO, min_obs, max_obs)
    
    # 2. Reparar Enemigos (usualmente entre 5 y 10)
    min_ene, max_ene = CONSTRAINTS[Elemento.ENEMIGO]
    mapa = reparar_cantidad_elemento(mapa, Elemento.ENEMIGO, min_ene, max_ene)
    
    # 3. Reparar Recompensas (usualmente entre 3 y 6)
    min_rec, max_rec = CONSTRAINTS[Elemento.RECOMPENZA]
    mapa = reparar_cantidad_elemento(mapa, Elemento.RECOMPENZA, min_rec, max_rec)

    # Oswaldo protege su vida: asegurar definitivamente que las esquinas estén intactas tras la reparación.
    mapa[0, 0] = Elemento.INICIO
    mapa[-1, -1] = Elemento.META

    return mapa

def update_particle_velocity_position(
    posicion_actual: NDArray[np.float64],
    velocidad_actual: NDArray[np.float64],
    mejor_posicion_individual: NDArray[np.float64],
    mejor_posicion_global: NDArray[np.float64],
    w: float = 0.5,
    c1: float = 1.5,
    c2: float = 1.5
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Actualiza la velocidad y la posición de una partícula basándose en la fórmula matemática del PSO.
    Retorna una tupla con (nueva_posicion, nueva_velocidad).
    """
    # 1. Generar aleatoriedad para la decisión de este turno
    r1 = rng.random(MAP_SHAPE)
    r2 = rng.random(MAP_SHAPE)

    # 2. Calcular los tres componentes de la velocidad
    inercia = w * velocidad_actual
    cognitivo = c1 * r1 * (mejor_posicion_individual - posicion_actual)
    social = c2 * r2 * (mejor_posicion_global - posicion_actual)

    # 3. Sumar para obtener la nueva velocidad (v_t+1)
    nueva_velocidad = inercia + cognitivo + social

    # 4. Actualizar la posición sumando la velocidad (x_t+1)
    nueva_posicion = posicion_actual + nueva_velocidad

    # 5. Restringir la posición para que los decimales no salgan de [0.0, 1.0]
    nueva_posicion = np.clip(nueva_posicion, 0.0, 1.0)

    # Proteger las esquinas (inicio y meta deben mantenerse en 0.0 decimal)
    nueva_posicion[0, 0] = 0.0
    nueva_posicion[-1, -1] = 0.0

    return nueva_posicion, nueva_velocidad

def evaluar_y_actualizar_mejores(
    posiciones_actuales: list[NDArray[np.float64]],
    mejores_posiciones_individuales: list[NDArray[np.float64]],
    mejores_fitness_individuales: list[float],
    mejor_posicion_global: NDArray[np.float64] | None,
    mejor_fitness_global: float,
    evaluador: calculate_fitness
) -> tuple[list[NDArray[np.float64]], list[float], NDArray[np.float64], float]:
    """
    Evalúa el mapa de cada partícula en el turno actual y actualiza
    tanto su mejor posición histórica individual como la mejor global del enjambre.
    """
    # Recorremos cada partícula del enjambre
    for i in range(len(posiciones_actuales)):
        particula_actual = posiciones_actuales[i]
        
        # 1. Convertir a mapa físico con los límites reparados (Tarea #27)
        mapa_fisico = particle_to_map(particula_actual)
        
        # 2. Calificar el mapa (el evaluador devuelve el fitness como primer valor)
        fitness_actual, _ = evaluador.evaluar_mapa(mapa_fisico)
        
        # 3. Actualizar memoria individual: ¿Es este mapa mejor que su mejor histórico?
        if fitness_actual < mejores_fitness_individuales[i]:
            mejores_fitness_individuales[i] = fitness_actual
            mejores_posiciones_individuales[i] = particula_actual.copy()
            
        # 4. Actualizar memoria global: ¿Es este mapa el mejor de todo el enjambre?
        if fitness_actual < mejor_fitness_global:
            mejor_fitness_global = fitness_actual
            mejor_posicion_global = particula_actual.copy()

    return mejores_posiciones_individuales, mejores_fitness_individuales, mejor_posicion_global, mejor_fitness_global

def ejecutar_pso(num_particulas: int, iteraciones: int) -> tuple[NDArray[np.int32], list[float]]:
    """Ejecuta el ciclo completo de optimización por enjambre de partículas."""
    # 1. Inicializar posiciones y velocidades aleatorias
    posiciones = [generate_random_particle() for _ in range(num_particulas)]
    velocidades = [np.zeros(MAP_SHAPE) for _ in range(num_particulas)]
    
    # 2. Inicializar memorias
    mejores_posiciones_individuales = posiciones.copy()
    mejores_fitness_individuales = [float('inf')] * num_particulas
    
    mejor_posicion_global = None
    mejor_fitness_global = float('inf')
    
    historial_fitness = []
    
    # 3. Bucle principal
    for _ in range(iteraciones):
        # Evaluar mapas y actualizar los récords (Tarea #29)
        mejores_posiciones_individuales, mejores_fitness_individuales, mejor_posicion_global, mejor_fitness_global = evaluar_y_actualizar_mejores(
            posiciones, mejores_posiciones_individuales, mejores_fitness_individuales, mejor_posicion_global, mejor_fitness_global
        )
        
        historial_fitness.append(mejor_fitness_global)
        
        # Calcular nueva velocidad y mover cada partícula (Tarea #28)
        for i in range(num_particulas):
            posiciones[i], velocidades[i] = update_particle_velocity_position(
                posiciones[i], velocidades[i], mejores_posiciones_individuales[i], mejor_posicion_global
            )
            
    # 4. Extraer el mejor resultado absoluto y convertirlo a enteros
    mejor_mapa_final = particle_to_map(mejor_posicion_global)
    return mejor_mapa_final, historial_fitness