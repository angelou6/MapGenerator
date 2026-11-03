"""
Ejecutor unificado de AG, PSO y RS.

Los tres reciben la MISMA poblacion inicial (lista de mapas + la Config de cada uno) y el
mismo presupuesto por individuo, asi la comparacion es justa:
  AG  : `presupuesto` generaciones      (~1 evaluacion por individuo por generacion)
  PSO : `presupuesto` iteraciones       (1 evaluacion por particula por iteracion)
  RS  : `presupuesto` pasos por mapa    (1 evaluacion por paso)
Cuando la poblacion mezcla presets (opcion "mezclar"), los mapas se agrupan por preset y
cada grupo se optimiza con sus propios rangos y dificultad objetivo.
"""
import math
import random
import time
from dataclasses import dataclass, field

import numpy as np

from generator.evaluacion import Config, Evaluador
from generator.filete import simulated_annealing
from generator.genetic import Genetic
from generator.map_utils import DEFAULT_WEIGHTS
from generator.particulas import ejecutar_pso

TEMP_INICIAL = 30.0
TEMP_FINAL = 0.3


@dataclass
class Resultado:
    nombre: str
    mapas: list                     # mapas finales
    pareo: str                      # "indice" (mapa i -> mapa i) o "ranking" (AG)
    tiempo: float
    evaluaciones: int
    mejor: np.ndarray = field(default_factory=lambda: np.array([]))   # por paso
    media: np.ndarray = field(default_factory=lambda: np.array([]))   # por paso


def _grupos(configs: list[Config]) -> dict[tuple, list[int]]:
    grupos: dict[tuple, list[int]] = {}
    for i, c in enumerate(configs):
        grupos.setdefault(c.clave, []).append(i)
    return grupos


def _combinar(historiales: list[tuple[int, np.ndarray, np.ndarray]]):
    """Une el historial de varios grupos: mejor = minimo, media = promedio ponderado."""
    total = sum(n for n, _, _ in historiales)
    mejor = np.min([m for _, m, _ in historiales], axis=0)
    media = sum(n * md for n, _, md in historiales) / total
    return mejor, media


class _Progreso:
    def __init__(self, etiqueta, total):
        self.etiqueta, self.total, self.hecho, self._ultimo = etiqueta, total, 0, -1

    def avanzar(self, n=1):
        self.hecho += n
        pct = int(100 * self.hecho / self.total)
        if pct != self._ultimo:  # solo se imprime cuando cambia el porcentaje entero
            self._ultimo = pct
            print(f"\r   {self.etiqueta}: {pct:3d}%", end="", flush=True)

    def fin(self):
        print(f"\r   {self.etiqueta}: 100%  ")


def ejecutar_ag(mapas, configs, pesos=DEFAULT_WEIGHTS, presupuesto=150, mostrar=True):
    t0 = time.perf_counter()
    prog = _Progreso("Algoritmo genético", presupuesto * len(_grupos(configs))) if mostrar else None
    finales = [None] * len(mapas)
    hists, evals = [], 0
    for idx in _grupos(configs).values():
        cfg = configs[idx[0]]
        ev = Evaluador(cfg, pesos)
        ag = Genetic(ev, cfg.rangos, [mapas[i] for i in idx])
        pob, hist = ag.entrenar(presupuesto, (lambda g, t: prog.avanzar()) if prog else None)
        evals += ev.llamadas
        # El AG produce una poblacion nueva (no hay correspondencia mapa a mapa). Para poder
        # comparar contra la inicial se parean por ranking de fitness dentro del mismo preset:
        # el mejor original con el mejor final, el segundo con el segundo, etc.
        orden_orig = sorted(idx, key=lambda i: ev.fitness(mapas[i]))
        orden_fin = sorted(pob, key=ev.fitness)
        for i, m in zip(orden_orig, orden_fin):
            finales[i] = m
        hists.append((len(idx), np.array([h["mejor"] for h in hist]),
                      np.array([h["media"] for h in hist])))
    if prog:
        prog.fin()
    mejor, media = _combinar(hists)
    return Resultado("AG", finales, "ranking", time.perf_counter() - t0, evals, mejor, media)


def ejecutar_pso_poblacion(mapas, configs, pesos=DEFAULT_WEIGHTS, presupuesto=150, mostrar=True):
    t0 = time.perf_counter()
    prog = _Progreso("PSO", presupuesto * len(_grupos(configs))) if mostrar else None
    finales = [None] * len(mapas)
    hists, evals = [], 0
    for idx in _grupos(configs).values():
        cfg = configs[idx[0]]
        ev = Evaluador(cfg, pesos)
        res, hist = ejecutar_pso(ev, cfg.rangos, [mapas[i] for i in idx], presupuesto,
                                 (lambda it, t: prog.avanzar()) if prog else None)
        for i, m in zip(idx, res):
            finales[i] = m
        hists.append((len(idx), np.array([h["mejor"] for h in hist]),
                      np.array([h["media"] for h in hist])))
        evals += ev.llamadas
    if prog:
        prog.fin()
    mejor, media = _combinar(hists)
    return Resultado("PSO", finales, "indice", time.perf_counter() - t0, evals, mejor, media)


def ejecutar_rs(mapas, configs, pesos=DEFAULT_WEIGHTS, presupuesto=150, mostrar=True):
    t0 = time.perf_counter()
    prog = _Progreso("Recocido simulado", len(mapas)) if mostrar else None
    enfriamiento = (TEMP_FINAL / TEMP_INICIAL) ** (1 / max(1, presupuesto))
    evaluadores: dict[tuple, Evaluador] = {}
    finales, hists_por_mapa = [], []
    for mapa, cfg in zip(mapas, configs):
        ev = evaluadores.setdefault(cfg.clave, Evaluador(cfg, pesos))
        hist: list[tuple[float, float]] = []
        finales.append(simulated_annealing(
            np.array(mapa), cfg.dificultad, steps=presupuesto, temp=TEMP_INICIAL,
            cooling_rate=enfriamiento, history=hist, weights=pesos, rangos=cfg.rangos,
            evaluar=ev.fitness))
        hists_por_mapa.append([b for _, b in hist])
        if prog:
            prog.avanzar()
    if prog:
        prog.fin()
    h = np.array(hists_por_mapa)
    return Resultado("RS", finales, "indice", time.perf_counter() - t0,
                     sum(e.llamadas for e in evaluadores.values()), h.min(axis=0), h.mean(axis=0))


ALGORITMOS = {"AG": ejecutar_ag, "PSO": ejecutar_pso_poblacion, "RS": ejecutar_rs}
