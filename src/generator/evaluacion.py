"""
Evaluador unico del proyecto.

Todos los algoritmos (AG, PSO, RS) y todas las graficas usan ESTA clase, asi el fitness
es siempre el f(X) del documento (se minimiza) y la definicion de "mapa viable" es la misma.
"""
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from generator.map_utils import (
    DEFAULT_WEIGHTS,
    calculate_dificulty_penalty,
    calculate_distribution_penalty,
    calculate_enemy_penalty,
    calculate_fitness,
    calculate_path_penalty,
    calculate_reward_penalty,
    get_difficulty,
)
from path_search.breadth import bfs
from path_search.utils import get_ajacent_blockers
from representation import MAP_SHAPE, Elemento

# Un mapa es viable si su dificultad queda a +-10 del objetivo: la mitad de la separacion
# entre niveles (30/50/70), asi las bandas facil/medio/dificil no se traslapan.
TOLERANCIA_DIFICULTAD = 10

# Escenario "Exploracion" (6-10 recompensas y CAMINO LARGO). El documento no fija cuanto
# es "largo"; se toma L >= 40 (el L del ejemplo del documento) y se penaliza lo que falte.
CAMINO_LARGO_MIN = 40
PESO_CAMINO_LARGO = 5


@dataclass(eq=False)
class Config:
    """Configuracion de un preset/escenario: rangos de elementos y dificultad objetivo."""

    nombre: str
    dificultad: int
    rangos: dict[Elemento, tuple[int, int]]
    camino_largo: bool = False

    @property
    def clave(self) -> tuple:
        return (self.nombre, self.dificultad, self.camino_largo,
                tuple(sorted((int(k), v) for k, v in self.rangos.items())))


@dataclass
class Reporte:
    """Todos los datos que el documento pide para describir un mapa."""

    viable: bool
    motivos: list[str]
    ruta: list[tuple[int, int]] | None
    longitud: int            # L
    enemigos_cerca: int      # E
    obstaculos_cerca: int    # P
    recompensas_acc: int     # R
    dificultad: float | None  # D = 5E + 0.5L + 2P - 4R
    dificultad_obj: int
    p_camino: float
    p_dificultad: float
    p_enemigos: float
    p_recompensas: float
    p_distribucion: float
    p_camino_largo: float
    fitness: float           # f(X)
    n_obstaculos: int
    n_enemigos: int
    n_recompensas: int


class Evaluador:
    """Calcula fitness y reportes completos para una Config y unos pesos dados."""

    def __init__(self, config: Config, pesos=DEFAULT_WEIGHTS):
        self.config = config
        self.pesos = tuple(pesos)
        self.llamadas = 0  # evaluaciones solicitadas (para medir costo computacional)
        self._cache: dict[bytes, float] = {}

    # ------------------------------------------------------------------ fitness
    def fitness(self, mapa: NDArray) -> float:
        """f(X) del documento (+ penalizacion de camino largo si el escenario la pide)."""
        self.llamadas += 1
        clave = np.ascontiguousarray(mapa, dtype=np.int8).tobytes()
        if clave in self._cache:
            return self._cache[clave]
        ruta = bfs(mapa)
        f = float(calculate_fitness(mapa, ruta, self.config.dificultad, self.pesos))
        f += self._penalizacion_largo(ruta)
        if len(self._cache) < 200_000:
            self._cache[clave] = f
        return f

    def _penalizacion_largo(self, ruta) -> float:
        if self.config.camino_largo and ruta is not None:
            return PESO_CAMINO_LARGO * max(0, CAMINO_LARGO_MIN - len(ruta))
        return 0.0

    # ------------------------------------------------------------------ reporte
    def reporte(self, mapa: NDArray) -> Reporte:
        cfg = self.config
        ruta = bfs(mapa)
        p_camino = float(calculate_path_penalty(mapa, ruta))
        p_ene = float(calculate_enemy_penalty(mapa))
        p_rec = float(calculate_reward_penalty(mapa))
        p_dis = float(calculate_distribution_penalty(mapa))
        p_largo = self._penalizacion_largo(ruta)

        if ruta is not None:
            bloq = get_ajacent_blockers(mapa, ruta)
            e, p = bloq[Elemento.ENEMIGO], bloq[Elemento.OBSTACULO]
            dif = float(get_difficulty(mapa, ruta))
            p_dif = float(calculate_dificulty_penalty(mapa, ruta, cfg.dificultad))
            from generator.map_utils import count_accesible_rewards
            r = int(count_accesible_rewards(mapa))
            largo = len(ruta)
        else:
            e = p = r = largo = 0
            dif, p_dif = None, 10000.0

        n_o = int(np.count_nonzero(mapa == Elemento.OBSTACULO))
        n_e = int(np.count_nonzero(mapa == Elemento.ENEMIGO))
        n_r = int(np.count_nonzero(mapa == Elemento.RECOMPENZA))

        motivos = []
        if ruta is None:
            motivos.append("sin camino")
        for el, n, nombre in ((Elemento.OBSTACULO, n_o, "obstaculos"),
                              (Elemento.ENEMIGO, n_e, "enemigos"),
                              (Elemento.RECOMPENZA, n_r, "recompensas")):
            lo, hi = cfg.rangos.get(el, (0, MAP_SHAPE[0] * MAP_SHAPE[1]))
            if not lo <= n <= hi:
                motivos.append(f"{nombre} fuera de rango")
        if (np.count_nonzero(mapa == Elemento.INICIO) != 1
                or np.count_nonzero(mapa == Elemento.META) != 1
                or mapa[0, 0] != Elemento.INICIO or mapa[-1, -1] != Elemento.META):
            motivos.append("inicio/meta alterados")
        if p_rec > 0:
            motivos.append("recompensa encerrada")
        if p_ene > 0:
            motivos.append("enemigos aglomerados")
        if dif is not None and abs(dif - cfg.dificultad) > TOLERANCIA_DIFICULTAD:
            motivos.append(f"dificultad fuera de +-{TOLERANCIA_DIFICULTAD}")
        if cfg.camino_largo and ruta is not None and len(ruta) < CAMINO_LARGO_MIN:
            motivos.append("camino corto")

        f = float(calculate_fitness(mapa, ruta, cfg.dificultad, self.pesos)) + p_largo
        return Reporte(
            viable=not motivos, motivos=motivos, ruta=ruta, longitud=largo,
            enemigos_cerca=int(e), obstaculos_cerca=int(p), recompensas_acc=r,
            dificultad=dif, dificultad_obj=cfg.dificultad, p_camino=p_camino,
            p_dificultad=p_dif, p_enemigos=p_ene, p_recompensas=p_rec,
            p_distribucion=p_dis, p_camino_largo=p_largo, fitness=f,
            n_obstaculos=n_o, n_enemigos=n_e, n_recompensas=n_r,
        )
