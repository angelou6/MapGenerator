"""
Modo lote (uv run map --lote): experimentos de las secciones 12, 13 y 17 del documento.

 3 algoritmos x 6 escenarios x N corridas independientes (N = 10 -> 180 corridas).
 Cada corrida toma una poblacion inicial NUEVA de `--poblacion-lote` mapas del escenario, la
 optimiza con el algoritmo (mismo presupuesto para todos) y registra:
   - mejor fitness de la corrida, fitness medio de la poblacion final, % de mapas viables,
     dificultad obtenida, diversidad (Hamming), tiempo y evaluaciones.
 Salidas (carpeta resultados/lote_<fecha>): corridas.csv, resumen_por_escenario.csv,
 resumen_por_algoritmo.csv, mejores_mapas.json, preguntas.txt y las graficas.
"""
import json
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import numpy as np

from generator.algoritmos import ALGORITMOS
from generator.evaluacion import Evaluador
from generator.map_utils import fijar_semilla, generate_random_map
from generator.presets import config_de_escenario
from generator.scenarios import SCENARIOS
from mapgenerator import estadisticas as est
from mapgenerator import graficas

ALGOS = list(ALGORITMOS)


def _una_corrida(tarea):
    algoritmo, escenario, corrida, n_pob, presupuesto, pesos, semilla = tarea
    fijar_semilla(semilla)
    cfg = config_de_escenario(escenario)
    mapas = [generate_random_map(cfg.rangos) for _ in range(n_pob)]
    res = ALGORITMOS[algoritmo](mapas, [cfg] * n_pob, pesos, presupuesto, mostrar=False)
    ev = Evaluador(cfg, pesos)
    reps = [ev.reporte(m) for m in res.mapas]
    st = est.estadisticas(algoritmo, res.mapas, reps, res.tiempo, res.evaluaciones)
    i = int(np.argmin([r.fitness for r in reps]))
    return {
        "algoritmo": algoritmo, "escenario": escenario, "corrida": corrida, "semilla": semilla,
        "mejor_fitness": st["mejor_fitness"], "fitness_medio": st["fitness_medio"],
        "pct_viables": st["pct_viables"], "mejor_es_viable": int(reps[i].viable),
        "dificultad_mejor": reps[i].dificultad, "dificultad_media": st["dificultad_media"],
        "dificultad_obj": cfg.dificultad, "longitud_mejor": reps[i].longitud,
        "enemigos_cerca_mejor": reps[i].enemigos_cerca, "diversidad": st["diversidad"],
        "tiempo": st["tiempo"], "evaluaciones": st["evaluaciones"],
        "curva_mejor": [float(x) for x in res.mejor],
        "mejor_mapa": np.asarray(res.mapas[i]).tolist(),
    }


def _stats(filas):
    mejor = np.array([f["mejor_fitness"] for f in filas])
    dif = [f["dificultad_mejor"] for f in filas if f["dificultad_mejor"] is not None]
    return {
        "mejor": float(mejor.min()), "peor": float(mejor.max()), "promedio": float(mejor.mean()),
        "desv": float(mejor.std()), "tiempo_prom": float(np.mean([f["tiempo"] for f in filas])),
        "pct_validos": float(np.mean([f["pct_viables"] for f in filas])),
        "pct_corridas_mejor_viable": 100.0 * float(np.mean([f["mejor_es_viable"] for f in filas])),
        "dificultad": float(np.mean(dif)) if dif else float("nan"),
        "diversidad": float(np.mean([f["diversidad"] for f in filas])),
        "corridas": len(filas),
    }


def _tabla(titulo, filas_tabla):
    cab = ["Algoritmo", "Mejor f", "Peor f", "f prom.", "Desv.", "% válidos", "Dif. obt.",
           "Diversidad", "Tiempo(s)"]
    datos = [[n, f"{s['mejor']:.1f}", f"{s['peor']:.1f}", f"{s['promedio']:.1f}", f"{s['desv']:.1f}",
              f"{s['pct_validos']:.0f}%", f"{s['dificultad']:.1f}", f"{s['diversidad']:.1f}",
              f"{s['tiempo_prom']:.1f}"] for n, s in filas_tabla]
    anchos = [max(len(c), *(len(f[i]) for f in datos)) for i, c in enumerate(cab)]
    lin = "  ".join(c.ljust(a) for c, a in zip(cab, anchos))
    return "\n".join([titulo, lin, "-" * len(lin)] + ["  ".join(v.ljust(a) for v, a in zip(f, anchos)) for f in datos])


def _preguntas(filas, por_algo, por_esc_algo):
    """Datos para las preguntas de la seccion 11 que se pueden responder con este lote."""
    L = []
    mejor_alg = min(por_algo, key=lambda a: por_algo[a]["promedio"])
    L.append(f"1. Menor fitness (promedio de mejores): {mejor_alg} "
             + ", ".join(f"{a}={por_algo[a]['promedio']:.1f}" for a in ALGOS))
    L.append("2. % de niveles válidos (población final): "
             + ", ".join(f"{a}={por_algo[a]['pct_validos']:.0f}%" for a in ALGOS))
    L.append("3. Convergencia: paso en que la curva del mejor fitness medio llega al 90% de su mejora total:")
    for a in ALGOS:
        curvas = np.array([f["curva_mejor"] for f in filas if f["algoritmo"] == a])
        c = np.mean(np.clip(curvas, 0, 1e4), axis=0)  # se acota para que los 600000 no dominen
        mejora = c[0] - c[-1]
        paso = int(np.argmax(c <= c[0] - 0.9 * mejora)) if mejora > 0 else 0
        L.append(f"     {a}: paso {paso} de {len(c) - 1}")
    L.append("4. Menor variabilidad (desv. del mejor fitness entre corridas): "
             + ", ".join(f"{a}={por_algo[a]['desv']:.1f}" for a in ALGOS))
    L.append("5. Mayor diversidad (Hamming medio): "
             + ", ".join(f"{a}={por_algo[a]['diversidad']:.1f}" for a in ALGOS))
    L.append("6. Dificultad 70 (escenario 'dificil') — % válidos / error de dificultad del mejor mapa:")
    for a in ALGOS:
        fs = [f for f in filas if f["algoritmo"] == a and f["escenario"] == "dificil"]
        err = [abs(f["dificultad_mejor"] - 70) for f in fs if f["dificultad_mejor"] is not None]
        L.append(f"     {a}: {np.mean([f['pct_viables'] for f in fs]):.0f}% válidos, "
                 f"error medio {np.mean(err) if err else float('nan'):.1f}")
    L.append("7-9. (obstáculos / recompensas / enemigos) requieren un barrido propio; el lote compara "
             "'laberinto' (más obstáculos), 'combate' (más enemigos) y 'exploracion' (más recompensas) "
             "contra 'medio' (mismo D objetivo = 50):")
    for esc in ("medio", "laberinto", "combate", "exploracion"):
        vals = [f for f in filas if f["escenario"] == esc]
        L.append(f"     {esc:12s} f prom. {np.mean([f['mejor_fitness'] for f in vals]):7.1f}  "
                 f"% válidos {np.mean([f['pct_viables'] for f in vals]):3.0f}%")
    conruta = [f for f in filas if f["dificultad_mejor"] is not None]
    r1 = np.corrcoef([f["longitud_mejor"] for f in conruta], [f["dificultad_mejor"] for f in conruta])[0, 1]
    r2 = np.corrcoef([f["enemigos_cerca_mejor"] for f in conruta], [f["dificultad_mejor"] for f in conruta])[0, 1]
    L.append(f"10. ¿PSO competitivo? Posición del PSO por f promedio: "
             f"{sorted(ALGOS, key=lambda a: por_algo[a]['promedio']).index('PSO') + 1} de 3")
    L.append(f"11. Correlación longitud del camino ↔ dificultad (mejores mapas): {r1:+.2f}")
    L.append(f"12. Correlación enemigos junto al camino ↔ dificultad (mejores mapas): {r2:+.2f}")
    return "\n".join(L)


def ejecutar_lote(args):
    escenarios = list(SCENARIOS)
    tareas = []
    base = args.semilla if args.semilla is not None else int(time.time()) % 100000
    for e_i, esc in enumerate(escenarios):
        for a_i, alg in enumerate(ALGOS):
            for c in range(args.corridas):
                semilla = base + e_i * 10000 + c  # misma poblacion inicial para los 3 algoritmos
                tareas.append((alg, esc, c, args.poblacion_lote, args.presupuesto, tuple(args.pesos), semilla))
    total = len(tareas)
    carpeta = args.salida / f"lote_{datetime.now():%Y%m%d_%H%M%S}"
    carpeta.mkdir(parents=True, exist_ok=True)
    print(f"Lote: {len(ALGOS)} algoritmos x {len(escenarios)} escenarios x {args.corridas} corridas = "
          f"{total} corridas ({args.poblacion_lote} mapas por corrida, presupuesto {args.presupuesto}).")

    filas, t0 = [], time.perf_counter()
    if args.procesos > 1:
        with ProcessPoolExecutor(max_workers=args.procesos) as ex:
            for i, fila in enumerate(ex.map(_una_corrida, tareas, chunksize=1), 1):
                filas.append(fila)
                print(f"\r   corrida {i}/{total}", end="", flush=True)
    else:
        for i, tarea in enumerate(tareas, 1):
            filas.append(_una_corrida(tarea))
            print(f"\r   corrida {i}/{total}  ({time.perf_counter() - t0:.0f} s)", end="", flush=True)
    print(f"\nLote terminado en {time.perf_counter() - t0:.0f} s.")

    # ---- corridas.csv
    import csv
    campos = [k for k in filas[0] if k not in ("curva_mejor", "mejor_mapa")]
    with open(carpeta / "corridas.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos, extrasaction="ignore")
        w.writeheader()
        w.writerows(filas)

    # ---- agregados
    por_esc_algo = {(e, a): _stats([f for f in filas if f["escenario"] == e and f["algoritmo"] == a])
                    for e in escenarios for a in ALGOS}
    por_algo = {a: _stats([f for f in filas if f["algoritmo"] == a]) for a in ALGOS}
    with open(carpeta / "resumen_por_escenario.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["escenario", "algoritmo"] + list(por_algo[ALGOS[0]]))
        for (e, a), s in por_esc_algo.items():
            w.writerow([e, a] + list(s.values()))
    with open(carpeta / "resumen_por_algoritmo.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["algoritmo"] + list(por_algo[ALGOS[0]]))
        for a, s in por_algo.items():
            w.writerow([a] + list(s.values()))

    texto = [_tabla(f"TABLA GLOBAL ({total} corridas)", [(a, por_algo[a]) for a in ALGOS])]
    for e in escenarios:
        texto.append("\n" + _tabla(f"Escenario: {e}", [(a, por_esc_algo[(e, a)]) for a in ALGOS]))

    # ---- ganador (mismo criterio que el modo interactivo, con los agregados del lote)
    stats_rank = [{"algoritmo": a, "fitness_medio": por_algo[a]["promedio"],
                   "pct_viables": por_algo[a]["pct_validos"],
                   "error_dificultad": float(np.mean([abs(f["dificultad_mejor"] - f["dificultad_obj"])
                                                      for f in filas if f["algoritmo"] == a
                                                      and f["dificultad_mejor"] is not None])),
                   "diversidad": por_algo[a]["diversidad"], "desv_fitness": por_algo[a]["desv"],
                   "tiempo": por_algo[a]["tiempo_prom"]} for a in ALGOS]
    tabla_rank, ganador = est.ranking(stats_rank)
    texto.append("\nRanking global (1 = mejor):\n" + est.tabla_ranking_texto(tabla_rank))
    texto.append(f"\n>>> MEJOR ALGORITMO DEL LOTE: {ganador}")
    preguntas = _preguntas(filas, por_algo, por_esc_algo)
    texto.append("\nDATOS PARA LAS PREGUNTAS DE LA SECCION 11\n" + preguntas)
    salida = "\n".join(texto)
    print("\n" + salida)
    (carpeta / "resumen.txt").write_text(salida, encoding="utf-8")
    (carpeta / "preguntas.txt").write_text(preguntas, encoding="utf-8")

    # ---- mejores mapas por algoritmo y escenario (con su ruta) + JSON
    mejores_json = {}
    for e in escenarios:
        mejores = {}
        cfg = config_de_escenario(e)
        ev = Evaluador(cfg, tuple(args.pesos))
        for a in ALGOS:
            f = min((x for x in filas if x["escenario"] == e and x["algoritmo"] == a),
                    key=lambda x: x["mejor_fitness"])
            m = np.array(f["mejor_mapa"])
            mejores[a] = (m, ev.reporte(m))
            mejores_json[f"{e}/{a}"] = {"mapa": f["mejor_mapa"], "fitness": f["mejor_fitness"]}
        graficas.guardar_mejores_mapas(mejores, carpeta / f"mejores_mapas_{e}.png",
                                       f"Escenario {e}: mejor mapa de cada algoritmo")
    (carpeta / "mejores_mapas.json").write_text(json.dumps(mejores_json), encoding="utf-8")

    # ---- convergencia global, boxplot por escenario
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    for ax, e in zip(axes.ravel(), escenarios):
        for a in ALGOS:
            curvas = np.array([f["curva_mejor"] for f in filas if f["escenario"] == e and f["algoritmo"] == a])
            ax.plot(np.median(curvas, axis=0), label=a, color=graficas.COLOR_ALGO[a], linewidth=2)
        ax.set_yscale("log")
        ax.set_title(f"Convergencia — {e} (mediana de {args.corridas} corridas)")
        ax.set_xlabel("Paso")
        ax.set_ylabel("Mejor f(X)")
        ax.grid(alpha=0.3, which="both")
        ax.legend()
    fig.tight_layout()
    fig.savefig(carpeta / "convergencia_por_escenario.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    for ax, e in zip(axes.ravel(), escenarios):
        datos = [[f["mejor_fitness"] for f in filas if f["escenario"] == e and f["algoritmo"] == a] for a in ALGOS]
        bp = ax.boxplot(datos, tick_labels=ALGOS, patch_artist=True)
        for p, a in zip(bp["boxes"], ALGOS):
            p.set_facecolor(graficas.COLOR_ALGO[a])
            p.set_alpha(0.7)
        ax.set_yscale("log")
        ax.set_title(f"Mejor fitness por corrida — {e}")
        ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(carpeta / "boxplot_por_escenario.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    x = np.arange(len(escenarios))
    for ax, clave, titulo in ((axes[0], "pct_validos", "% de mapas válidos"),
                              (axes[1], "diversidad", "Diversidad (Hamming medio)"),
                              (axes[2], "tiempo_prom", "Tiempo promedio por corrida (s)")):
        for k, a in enumerate(ALGOS):
            ax.bar(x + (k - 1) * 0.27, [por_esc_algo[(e, a)][clave] for e in escenarios], 0.27,
                   label=a, color=graficas.COLOR_ALGO[a])
        ax.set_xticks(x, escenarios, rotation=20)
        ax.set_title(titulo)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(carpeta / "comparacion_por_escenario.png", dpi=120)
    plt.close(fig)
    print(f"\nTodo se guardó en: {carpeta.resolve()}")
