"""Estadisticas comparativas (tabla del documento, seccion 13) y ranking de algoritmos."""
import csv
from pathlib import Path

import numpy as np

from generator.map_utils import calculate_average_diversity

# criterio -> (clave, True si "menos es mejor"). Son los criterios que pide la conclusion
# del documento (calidad, dificultad, diversidad, estabilidad y costo) mas el % de validez.
CRITERIOS = {
    "Fitness medio": ("fitness_medio", True),
    "% viables": ("pct_viables", False),
    "Error de dificultad": ("error_dificultad", True),
    "Diversidad": ("diversidad", False),
    "Estabilidad (desv. fitness)": ("desv_fitness", True),
    "Tiempo": ("tiempo", True),
}


def estadisticas(nombre, mapas, reportes, tiempo=0.0, evaluaciones=0):
    fits = np.array([r.fitness for r in reportes])
    con_ruta = [r for r in reportes if r.dificultad is not None]
    return {
        "algoritmo": nombre,
        "mejor_fitness": float(fits.min()),
        "peor_fitness": float(fits.max()),
        "fitness_medio": float(fits.mean()),
        "fitness_mediana": float(np.median(fits)),
        "desv_fitness": float(fits.std()),
        "pct_viables": 100.0 * sum(r.viable for r in reportes) / len(reportes),
        "dificultad_media": float(np.mean([r.dificultad for r in con_ruta])) if con_ruta else float("nan"),
        "error_dificultad": float(np.mean([abs(r.dificultad - r.dificultad_obj) for r in con_ruta]))
        if con_ruta else float("nan"),
        "diversidad": calculate_average_diversity([np.asarray(m) for m in mapas]),
        "tiempo": float(tiempo),
        "evaluaciones": int(evaluaciones),
    }


def ranking(stats_algoritmos):
    """
    Posicion (1 = mejor) de cada algoritmo en cada criterio; el ganador es el de menor
    posicion promedio (todos los criterios pesan igual; desempate: fitness medio).
    Devuelve (lista ordenada de (nombre, promedio, {criterio: posicion}), ganador).
    """
    pos = {s["algoritmo"]: {} for s in stats_algoritmos}
    for criterio, (clave, menos_mejor) in CRITERIOS.items():
        orden = sorted(stats_algoritmos, key=lambda s: s[clave], reverse=not menos_mejor)
        for p, s in enumerate(orden, 1):
            pos[s["algoritmo"]][criterio] = p
    tabla = []
    for s in stats_algoritmos:
        n = s["algoritmo"]
        tabla.append((n, float(np.mean(list(pos[n].values()))), pos[n], s["fitness_medio"]))
    tabla.sort(key=lambda t: (t[1], t[3]))
    return [(n, m, p) for n, m, p, _ in tabla], tabla[0][0]


def tabla_texto(stats):
    """Tabla de la seccion 13 del documento (+ columnas extra utiles)."""
    cab = ["Algoritmo", "Mejor f", "Peor f", "f promedio", "Desv.", "% válidos",
           "Dif. obtenida", "Err. dif.", "Diversidad", "Tiempo(s)"]
    filas = [[s["algoritmo"], f"{s['mejor_fitness']:.1f}", f"{s['peor_fitness']:.1f}",
              f"{s['fitness_medio']:.1f}", f"{s['desv_fitness']:.1f}", f"{s['pct_viables']:.0f}%",
              f"{s['dificultad_media']:.1f}", f"{s['error_dificultad']:.1f}",
              f"{s['diversidad']:.1f}", f"{s['tiempo']:.1f}" if s["tiempo"] else "-"] for s in stats]
    anchos = [max(len(c), *(len(f[i]) for f in filas)) for i, c in enumerate(cab)]
    linea = "  ".join(c.ljust(a) for c, a in zip(cab, anchos))
    out = [linea, "-" * len(linea)]
    out += ["  ".join(v.ljust(a) for v, a in zip(f, anchos)) for f in filas]
    return "\n".join(out)


def tabla_ranking_texto(tabla):
    cab = ["Algoritmo", "Posición prom."] + list(CRITERIOS)
    filas = [[n, f"{m:.2f}"] + [str(p[c]) for c in CRITERIOS] for n, m, p in tabla]
    anchos = [max(len(c), *(len(f[i]) for f in filas)) for i, c in enumerate(cab)]
    linea = "  ".join(c.ljust(a) for c, a in zip(cab, anchos))
    return "\n".join([linea, "-" * len(linea)] + ["  ".join(v.ljust(a) for v, a in zip(f, anchos)) for f in filas])


def guardar_csv(stats, ruta: Path):
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(stats[0]))
        w.writeheader()
        w.writerows(stats)
