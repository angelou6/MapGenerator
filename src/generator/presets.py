"""
Presets de mapas: los 6 escenarios del documento (seccion 12) + personalizado + mezcla.

Los rangos por escenario viven en scenarios.py (SCENARIOS); aqui solo se arman las Config
que usan los algoritmos y el menu del main.
"""
import random

from generator.evaluacion import Config
from generator.map_utils import CONSTRAINTS
from generator.scenarios import SCENARIOS, get_scenario
from representation import Elemento

# El documento solo fija D objetivo para facil/medio/dificil. Para laberinto, combate y
# exploracion se usa un nivel medio por defecto (el usuario puede cambiarlo al elegir).
DIFICULTAD_ESPECIALES = 50

NOMBRES_BONITOS = {
    "facil": "Fácil", "medio": "Medio", "dificil": "Difícil",
    "laberinto": "Laberinto", "combate": "Combate", "exploracion": "Exploración",
}


def config_de_escenario(nombre: str, dificultad: int | None = None) -> Config:
    sc = get_scenario(nombre)
    if dificultad is None:
        dificultad = sc["dificultad"] if sc["dificultad"] is not None else DIFICULTAD_ESPECIALES
    return Config(nombre, int(dificultad), dict(sc["rangos"]), sc["camino_largo"])


def config_personalizada(obstaculos: tuple[int, int], enemigos: tuple[int, int],
                         recompensas: tuple[int, int], dificultad: int) -> Config:
    rangos = {Elemento.OBSTACULO: obstaculos, Elemento.ENEMIGO: enemigos,
              Elemento.RECOMPENZA: recompensas}
    total = sum(hi for _, hi in rangos.values())
    if any(lo > hi or lo < 0 for lo, hi in rangos.values()):
        raise ValueError("Cada rango debe cumplir 0 <= minimo <= maximo")
    if total > 223:
        raise ValueError("Los maximos suman mas de las 223 casillas libres del mapa")
    return Config("personalizado", int(dificultad), rangos)


def configs_mixtas(n: int) -> list[Config]:
    """Reparte los n mapas en partes iguales entre los 6 escenarios (orden aleatorio)."""
    nombres = list(SCENARIOS)
    asignados = [nombres[i % len(nombres)] for i in range(n)]
    random.shuffle(asignados)
    cache = {nom: config_de_escenario(nom) for nom in nombres}
    return [cache[nom] for nom in asignados]  # mismo objeto Config por escenario


def rangos_por_defecto() -> dict[Elemento, tuple[int, int]]:
    return dict(CONSTRAINTS)
