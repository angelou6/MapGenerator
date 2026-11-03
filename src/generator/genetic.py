"""
Algoritmo Genetico (AG) para mapas de 15x15.

Cromosoma: el mapa completo (225 genes; inicio y meta fijos).
Seleccion por torneo, cruza por regiones de 5x5, mutacion de una celda, reparacion de
rangos y elitismo. El fitness viene del Evaluador compartido: f(X) del documento (minimiza).
"""
import random

import numpy as np

from generator.map_utils import reparar_cantidad_elemento
from representation import MAP_SHAPE, Elemento

# Las 9 regiones de 5x5 se numeran 0..8 (fila*3 + columna). La 0 contiene el inicio y la 8
# la meta, por eso nunca se intercambian.
REGIONES_INTERCAMBIABLES = [1, 2, 3, 4, 5, 6, 7]


class Genetic:
    def __init__(self, evaluador, rangos, poblacion_inicial, prob_mutacion=0.2,
                 tam_torneo=3, n_elite=1):
        """
        Args:
            evaluador: Evaluador (generator.evaluacion) con la Config del escenario.
            rangos: {Elemento: (min, max)} que debe respetar cada hijo.
            poblacion_inicial: lista de mapas (arrays 15x15).
            prob_mutacion: probabilidad de que un hijo mute una celda.
            tam_torneo: participantes por torneo.
            n_elite: mejores mapas que pasan intactos a la siguiente generacion.
        """
        self.evaluador = evaluador
        self.rangos = rangos
        self.prob_mutacion = prob_mutacion
        self.tam_torneo = tam_torneo
        self.n_elite = n_elite
        self.generation = 0
        self.population = [{"matrix": np.array(m, dtype=np.int32), "fitness": None}
                           for m in poblacion_inicial]
        self.population_size = len(self.population)
        self.historial: list[dict] = []

    def evaluar_poblacion(self):
        """Califica solo a los individuos nuevos (fitness None)."""
        for ind in self.population:
            if ind["fitness"] is None:
                ind["fitness"] = self.evaluador.fitness(ind["matrix"])

    def torneo_seleccion(self):
        """Elige tam_torneo individuos al azar y regresa el de menor fitness."""
        k = self.tam_torneo
        if len(self.population) >= k:
            participantes = random.sample(self.population, k)
        else:
            participantes = random.choices(self.population, k=k)
        return min(participantes, key=lambda ind: ind["fitness"])

    def cruzar_padres(self, padre1, padre2):
        """Hijo = copia del padre1 con 2-4 regiones de 5x5 tomadas del padre2."""
        hijo = np.copy(padre1["matrix"])
        for region in random.sample(REGIONES_INTERCAMBIABLES, random.randint(2, 4)):
            f0, c0 = (region // 3) * 5, (region % 3) * 5
            hijo[f0:f0 + 5, c0:c0 + 5] = padre2["matrix"][f0:f0 + 5, c0:c0 + 5]
        return {"matrix": hijo, "fitness": None}

    def mutar_hijo(self, hijo):
        """Con probabilidad prob_mutacion cambia una celda (no inicio/meta) a 0, 1, 2 o 3."""
        if random.random() < self.prob_mutacion:
            matriz = hijo["matrix"]
            while True:
                f, c = random.randrange(MAP_SHAPE[0]), random.randrange(MAP_SHAPE[1])
                if (f, c) not in ((0, 0), (MAP_SHAPE[0] - 1, MAP_SHAPE[1] - 1)):
                    break
            opciones = [e for e in (Elemento.TRANSITABLE, Elemento.OBSTACULO,
                                    Elemento.ENEMIGO, Elemento.RECOMPENZA)
                        if e != matriz[f, c]]
            matriz[f, c] = random.choice(opciones)
        return hijo

    def reparar_hijo(self, hijo):
        """Ajusta las cantidades de cada elemento a los rangos del escenario."""
        for elemento, (minimo, maximo) in self.rangos.items():
            reparar_cantidad_elemento(hijo["matrix"], elemento, minimo, maximo)
        return hijo

    def generar_siguiente_poblacion(self):
        """Elitismo + (cruza -> mutacion -> reparacion) hasta completar la poblacion."""
        ordenada = sorted(self.population, key=lambda ind: ind["fitness"])
        nueva = [{"matrix": np.copy(ind["matrix"]), "fitness": ind["fitness"]}
                 for ind in ordenada[:self.n_elite]]
        while len(nueva) < self.population_size:
            hijo = self.cruzar_padres(self.torneo_seleccion(), self.torneo_seleccion())
            nueva.append(self.reparar_hijo(self.mutar_hijo(hijo)))
        self.population = nueva
        self.generation += 1

    def _registrar(self):
        fits = [ind["fitness"] for ind in self.population]
        self.historial.append({"generacion": self.generation, "mejor": min(fits),
                               "media": float(np.mean(fits)), "peor": max(fits)})

    def entrenar(self, total_generaciones=100, progreso=None):
        """
        Evoluciona la poblacion.

        Returns:
            (mapas_finales, historial): lista de mapas de la poblacion final y la lista
            de dicts {generacion, mejor, media, peor} (generacion 0 = poblacion inicial).
        """
        self.evaluar_poblacion()
        self._registrar()
        for _ in range(total_generaciones):
            self.generar_siguiente_poblacion()
            self.evaluar_poblacion()
            self._registrar()
            if progreso:
                progreso(self.generation, total_generaciones)
        return [ind["matrix"] for ind in self.population], self.historial
