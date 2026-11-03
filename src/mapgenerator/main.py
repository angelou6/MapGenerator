"""
Flujo principal del proyecto (uv run map):

 1. Pide la poblacion inicial (recomendado > 100).
 2. Pide el preset de mapas (escenarios, personalizado o mezcla de todos).
 3. Muestra TODOS los mapas iniciales en una cuadricula de 3 columnas con sus datos.
 4. [ENTER] AG -> imagen original vs resultado.   5. [ENTER] PSO -> idem.   6. [ENTER] RS -> idem.
 7. Resultados finales: tabla, ranking, mejor algoritmo, convergencia, resumen y mejores mapas.

Para correr los 180 experimentos del documento (3 algoritmos x 6 escenarios x 10 corridas):
 uv run map --lote
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from generator.algoritmos import ALGORITMOS
from generator.evaluacion import TOLERANCIA_DIFICULTAD, Evaluador
from generator.map_utils import DEFAULT_WEIGHTS, fijar_semilla, generate_random_map
from generator.presets import (
    DIFICULTAD_ESPECIALES, NOMBRES_BONITOS, config_de_escenario,
    config_personalizada, configs_mixtas, rangos_por_defecto,
)
from generator.scenarios import SCENARIOS
from mapgenerator import estadisticas as est
from mapgenerator import graficas
from representation import Elemento

NOMBRES_ALGO = {"AG": "Algoritmo Genético", "PSO": "PSO (enjambre de partículas)",
                "RS": "Recocido Simulado"}
MENU_PRESETS = [
    ("1", "facil", "Fácil        (dificultad objetivo 30)"),
    ("2", "medio", "Medio        (dificultad objetivo 50)"),
    ("3", "dificil", "Difícil      (dificultad objetivo 70)"),
    ("4", "laberinto", "Laberinto    (35-50 obstáculos)"),
    ("5", "combate", "Combate      (10-15 enemigos)"),
    ("6", "exploracion", "Exploración  (6-10 recompensas y camino largo)"),
    ("7", "personalizado", "Personalizado (tú eliges cantidades de obstáculos, enemigos y recompensas)"),
    ("8", "mixto", "Mezclar todos los presets (la población se reparte entre los 6 escenarios)"),
]


# ----------------------------------------------------------------------------- entradas
def pedir_entero(mensaje, defecto=None, minimo=None, maximo=None):
    while True:
        txt = input(mensaje).strip()
        if txt == "" and defecto is not None:
            return defecto
        try:
            valor = int(txt)
        except ValueError:
            print("   Escribe un número entero.")
            continue
        if (minimo is not None and valor < minimo) or (maximo is not None and valor > maximo):
            print(f"   El valor debe estar entre {minimo} y {maximo}.")
            continue
        return valor


def pedir_poblacion(args):
    if args.poblacion:
        return args.poblacion
    print("\n=== POBLACIÓN INICIAL ===")
    print("Se recomienda una población MAYOR a 100 mapas (más diversidad para los algoritmos).")
    while True:
        n = pedir_entero("¿De cuánto quieres la población inicial? [150]: ", 150, minimo=6)
        if n > 100:
            return n
        if input(f"   {n} no supera los 100 recomendados. ¿Continuar igual? (s/n): ").strip().lower() == "s":
            return n


def pedir_preset(args, n):
    """Regresa (nombre_del_preset, lista de Config, una por mapa)."""
    nombre = args.preset
    if nombre is None:
        print("\n=== PRESET DE MAPAS ===")
        for clave, _, texto in MENU_PRESETS:
            print(f"  {clave}) {texto}")
        opciones = {c: nom for c, nom, _ in MENU_PRESETS}
        while True:
            eleccion = input("Elige una opción [2]: ").strip() or "2"
            if eleccion in opciones:
                nombre = opciones[eleccion]
                break
            print("   Opción no válida.")

    if nombre == "mixto":
        return "mixto", configs_mixtas(n)
    if nombre == "personalizado":
        d = rangos_por_defecto()
        while True:
            print("Define el rango (mínimo y máximo) de cada elemento; ENTER = valor del documento.")
            try:
                rangos = []
                for el, etiqueta in ((Elemento.OBSTACULO, "obstáculos"), (Elemento.ENEMIGO, "enemigos"),
                                     (Elemento.RECOMPENZA, "recompensas")):
                    lo = pedir_entero(f"   {etiqueta} mínimo [{d[el][0]}]: ", d[el][0], minimo=0)
                    hi = pedir_entero(f"   {etiqueta} máximo [{d[el][1]}]: ", d[el][1], minimo=0)
                    rangos.append((lo, hi))
                dif = args.dificultad or pedir_entero(
                    f"   dificultad objetivo [{DIFICULTAD_ESPECIALES}]: ", DIFICULTAD_ESPECIALES, minimo=0)
                cfg = config_personalizada(*rangos, dif)
                break
            except ValueError as e:
                print(f"   {e}. Intenta de nuevo.")
        return "personalizado", [cfg] * n

    dificultad = args.dificultad
    if dificultad is None and nombre in ("laberinto", "combate", "exploracion"):
        dificultad = pedir_entero(
            f"Dificultad objetivo (30 fácil / 50 medio / 70 difícil) [{DIFICULTAD_ESPECIALES}]: ",
            DIFICULTAD_ESPECIALES, minimo=0) if not args.sin_pausas else None
    return nombre, [config_de_escenario(nombre, dificultad)] * n


def pausa(mensaje, args):
    if args.sin_pausas:
        print(f"\n{mensaje}")
    else:
        input(f"\n{mensaje}")


# ----------------------------------------------------------------------------- evaluacion
class Evaluadores:
    """Un Evaluador por preset, para calificar y reportar cualquier mapa con su Config."""

    def __init__(self, pesos):
        self.pesos, self._ev = pesos, {}

    def de(self, cfg):
        return self._ev.setdefault(cfg.clave, Evaluador(cfg, self.pesos))

    def reportes(self, mapas, configs):
        return [self.de(c).reporte(m) for m, c in zip(mapas, configs)]


def resumen_corto(nombre, st):
    print(f"   {nombre}: viables {st['pct_viables']:.0f}%  |  mejor f {st['mejor_fitness']:.1f}  |  "
          f"f medio {st['fitness_medio']:.1f}  |  error dif. {st['error_dificultad']:.1f}  |  "
          f"diversidad {st['diversidad']:.1f}")


def correlacion(x, y):
    return float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 and np.std(x) > 0 and np.std(y) > 0 else float("nan")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Generación procedural de niveles: AG vs PSO vs RS")
    ap.add_argument("--pesos", type=float, nargs=5, default=DEFAULT_WEIGHTS,
                    metavar=("W1", "W2", "W3", "W4", "W5"), help="Pesos del fitness (50 10 5 5 2)")
    ap.add_argument("--presupuesto", type=int, default=150,
                    help="Generaciones (AG) = iteraciones (PSO) = pasos por mapa (RS). Default 150")
    ap.add_argument("--poblacion", type=int, help="Salta la pregunta de población")
    ap.add_argument("--preset", choices=list(SCENARIOS) + ["personalizado", "mixto"],
                    help="Salta el menú de presets")
    ap.add_argument("--dificultad", type=int, help="Dificultad objetivo (sobrescribe la del preset)")
    ap.add_argument("--semilla", type=int, help="Semilla para repetir exactamente una ejecución")
    ap.add_argument("--salida", type=Path, default=Path("resultados"), help="Carpeta de resultados")
    ap.add_argument("--no-abrir", action="store_true", help="Solo guardar las imágenes, no abrirlas")
    ap.add_argument("--sin-pausas", action="store_true", help="No esperar ENTER entre etapas")
    ap.add_argument("--lote", action="store_true",
                    help="Modo experimentos: 3 algoritmos x 6 escenarios x N corridas (180 por defecto)")
    ap.add_argument("--corridas", type=int, default=10, help="(--lote) corridas por algoritmo y escenario")
    ap.add_argument("--poblacion-lote", type=int, default=20, help="(--lote) mapas por corrida")
    ap.add_argument("--procesos", type=int, default=1, help="(--lote) procesos en paralelo")
    args = ap.parse_args()
    pesos = tuple(args.pesos)
    if any(p < 0 for p in pesos):
        ap.error("Los pesos no pueden ser negativos")
    if args.semilla is not None:
        fijar_semilla(args.semilla)

    if args.lote:
        from mapgenerator.lote import ejecutar_lote
        return ejecutar_lote(args)

    abrir = not args.no_abrir
    carpeta = args.salida / f"sesion_{datetime.now():%Y%m%d_%H%M%S}"
    carpeta.mkdir(parents=True, exist_ok=True)

    print("=" * 62)
    print("  GENERADOR PROCEDURAL DE NIVELES  ·  AG vs PSO vs Recocido")
    print("=" * 62)
    n = pedir_poblacion(args)
    nombre_preset, configs = pedir_preset(args, n)
    evs = Evaluadores(pesos)

    # ------------------------------------------------------------ población inicial
    print(f"\nGenerando {n} mapas iniciales ({nombre_preset})...")
    originales = [generate_random_map(c.rangos) for c in configs]
    rep_orig = evs.reportes(originales, configs)
    st_ini = est.estadisticas("Inicial", originales, rep_orig)
    print(f"Mapas viables en la generación inicial: {sum(r.viable for r in rep_orig)}/{n} "
          f"(viable = camino válido, cantidades en rango, sin recompensas encerradas ni enemigos "
          f"aglomerados y dificultad a ±{TOLERANCIA_DIFICULTAD} del objetivo; el camino largo de "
          f"Exploración se penaliza en el fitness pero no descalifica)")
    resumen_corto("Inicial", st_ini)
    print("Dibujando la cuadrícula de mapas iniciales...")
    paginas = graficas.guardar_cuadricula_paginas(
        originales, rep_orig, configs, f"Población inicial ({n} mapas, preset: {nombre_preset})",
        carpeta, "01_poblacion_inicial")
    ruta = carpeta / "01_poblacion_inicial.html"
    graficas.guardar_galeria(paginas, f"Población inicial ({n} mapas)", ruta)
    graficas.abrir(ruta, abrir)

    # ------------------------------------------------------------ los tres algoritmos
    resultados, reportes_fin, stats = {}, {}, [st_ini]
    for k, (clave, funcion) in enumerate(ALGORITMOS.items(), start=2):
        pausa(f"Presiona ENTER para pasar la población por: {NOMBRES_ALGO[clave]}...", args)
        res = funcion(originales, configs, pesos, args.presupuesto)
        rep = evs.reportes(res.mapas, configs)
        st = est.estadisticas(clave, res.mapas, rep, res.tiempo, res.evaluaciones)
        resultados[clave], reportes_fin[clave] = res, rep
        stats.append(st)
        print(f"   Tiempo: {res.tiempo:.1f} s · evaluaciones: {res.evaluaciones}")
        resumen_corto(clave, st)
        prefijo = f"{k:02d}_comparacion_{clave}"
        paginas = graficas.guardar_comparacion_paginas(
            originales, rep_orig, res.mapas, rep, configs, clave, res.pareo, carpeta, prefijo)
        ruta = carpeta / f"{prefijo}.html"
        graficas.guardar_galeria(paginas, f"{clave}: original → resultado", ruta)
        graficas.abrir(ruta, abrir)

    # ------------------------------------------------------------ resultados finales
    pausa("Presiona ENTER para ver los resultados finales...", args)
    algos = [s for s in stats if s["algoritmo"] != "Inicial"]
    tabla, ganador = est.ranking(algos)
    print("\n" + "=" * 62 + "\n  RESULTADOS FINALES\n" + "=" * 62)
    print(est.tabla_texto(stats))
    print("\nRanking (1 = mejor en ese criterio; gana la menor posición promedio):")
    print(est.tabla_ranking_texto(tabla))
    print(f"\n>>> MEJOR ALGORITMO: {ganador} ({NOMBRES_ALGO[ganador]})")

    # relaciones pedidas en el documento (preguntas 11 y 12) sobre todos los mapas evaluados
    todos = list(rep_orig) + [r for rs in reportes_fin.values() for r in rs]
    con_ruta = [r for r in todos if r.dificultad is not None]
    print(f"\nCorrelación longitud del camino ↔ dificultad: "
          f"{correlacion([r.longitud for r in con_ruta], [r.dificultad for r in con_ruta]):+.2f}")
    print(f"Correlación enemigos junto al camino ↔ dificultad: "
          f"{correlacion([r.enemigos_cerca for r in con_ruta], [r.dificultad for r in con_ruta]):+.2f}")

    fits = {"Inicial": [r.fitness for r in rep_orig]}
    fits.update({c: [r.fitness for r in rs] for c, rs in reportes_fin.items()})
    mejores = {}
    for c, rs in reportes_fin.items():
        i = int(np.argmin([r.fitness for r in rs]))
        mejores[c] = (resultados[c].mapas[i], rs[i])

    graficas.guardar_convergencia(resultados, carpeta / "05_convergencia.png")
    graficas.guardar_resumen(stats, fits, carpeta / "06_resumen.png")
    graficas.guardar_mejores_mapas(mejores, carpeta / "07_mejores_mapas.png")
    est.guardar_csv(stats, carpeta / "estadisticas.csv")
    with open(carpeta / "mapas.json", "w", encoding="utf-8") as f:
        json.dump({"preset": nombre_preset, "pesos": pesos, "presupuesto": args.presupuesto,
                   "inicial": [m.tolist() for m in originales],
                   **{c: [np.asarray(m).tolist() for m in r.mapas] for c, r in resultados.items()}}, f)
    for nombre in ("05_convergencia", "06_resumen", "07_mejores_mapas"):
        graficas.abrir(carpeta / f"{nombre}.png", abrir)
    print(f"\nTodo se guardó en: {carpeta.resolve()}")
    print("¡Ejecución completa!")


if __name__ == "__main__":
    main()