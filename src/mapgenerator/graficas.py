"""Imagenes del proyecto: cuadriculas de mapas, comparaciones, convergencia y resumen."""
import math
import webbrowser
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # las imagenes se guardan como PNG y se abren en el visor/navegador
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from generator.evaluacion import CAMINO_LARGO_MIN
from representation import Elemento

COLORES = {
    Elemento.TRANSITABLE: "#f4efe1", Elemento.OBSTACULO: "#4a4e57",
    Elemento.ENEMIGO: "#d64545", Elemento.RECOMPENZA: "#f2b705",
    Elemento.INICIO: "#2e9e5b", Elemento.META: "#2f6fdb",
}
CMAP = ListedColormap([COLORES[Elemento(i)] for i in range(6)])
COLOR_ALGO = {"Inicial": "#7f7f7f", "AG": "#1f77b4", "PSO": "#2ca02c", "RS": "#d95f02"}
VERDE, ROJO = "#1e8e3e", "#c62828"


def abrir(ruta: Path, activo: bool = True):
    """Abre la imagen en el visor/navegador predeterminado (se puede hacer zoom y scroll)."""
    print(f"   Imagen guardada: {ruta}")
    if activo:
        try:
            webbrowser.open(Path(ruta).resolve().as_uri())
        except Exception:
            pass


def _fmt(x, dec=1):
    if x is None:
        return "n/d"
    return f"{x:.2e}" if abs(x) >= 1e5 else f"{x:.{dec}f}"


def dibujar_mapa(ax, mapa, ruta=None, viable=None):
    ax.imshow(mapa, cmap=CMAP, vmin=0, vmax=5, interpolation="nearest")
    n = mapa.shape[0]
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="#cfc8b4", linewidth=0.4)
    ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
    if ruta:
        ys, xs = zip(*ruta)
        ax.plot(xs, ys, color="#8e24aa", linewidth=1.6, alpha=0.9, zorder=3)
    for letra, (f, c) in (("I", (0, 0)), ("M", (n - 1, n - 1))):
        ax.text(c, f, letra, ha="center", va="center", color="white",
                fontsize=8, fontweight="bold", zorder=4)
    if viable is not None:
        for sp in ax.spines.values():
            sp.set_edgecolor(VERDE if viable else ROJO)
            sp.set_linewidth(2.6)


def _lineas(r):
    dif = "D = n/d (sin camino)" if r.dificultad is None else \
        f"D = {r.dificultad:.1f}  (objetivo {r.dificultad_obj}, desv. {abs(r.dificultad - r.dificultad_obj):.1f})"
    largo = (f"  largo(≥{CAMINO_LARGO_MIN}): {'sí' if r.largo_ok else 'no'}" if r.exige_largo else "")
    return [
        f"L={r.longitud}  E={r.enemigos_cerca}  P={r.obstaculos_cerca}  R={r.recompensas_acc}{largo}",
        dif,
        f"Obst {r.n_obstaculos} · Enem {r.n_enemigos} · Rec {r.n_recompensas}",
        f"P_cam {_fmt(r.p_camino, 0)} · P_dif {_fmt(r.p_dificultad)} · P_ene {_fmt(r.p_enemigos, 0)}",
        f"P_rec {_fmt(r.p_recompensas, 0)} · P_dis {_fmt(r.p_distribucion, 0)}"
        + (f" · P_largo {_fmt(r.p_camino_largo, 0)}" if r.p_camino_largo else ""),
        f"f(X) = {_fmt(r.fitness)}",
    ]


def _estado(r, extra=""):
    if r.viable:
        return "VIABLE", VERDE
    return "NO VIABLE", ROJO


def _motivo(r):
    return "; ".join(r.motivos[:2]) + ("…" if len(r.motivos) > 2 else "") if r.motivos else ""


def _lienzo(n_celdas, columnas, w, h, cabecera):
    filas = math.ceil(n_celdas / columnas)
    ancho, alto = columnas * w, filas * h + cabecera
    dpi = int(max(40, min(90, 14000 / alto)))
    fig = plt.figure(figsize=(ancho, alto), dpi=dpi)
    return fig, filas, ancho, alto


def _leyenda(fig, x, y):
    etiquetas = [("Libre", Elemento.TRANSITABLE), ("Obstáculo", Elemento.OBSTACULO),
                 ("Enemigo", Elemento.ENEMIGO), ("Recompensa", Elemento.RECOMPENZA),
                 ("Inicio", Elemento.INICIO), ("Meta", Elemento.META)]
    h = [Patch(facecolor=COLORES[e], edgecolor="#888", label=n) for n, e in etiquetas]
    h.append(plt.Line2D([], [], color="#8e24aa", linewidth=1.6, label="Camino BFS"))
    fig.legend(handles=h, loc="upper center", ncol=7, bbox_to_anchor=(x, y), frameon=False, fontsize=9)


def guardar_cuadricula(mapas, reportes, configs, titulo, ruta, columnas=3, inicio=0, extra=None):
    """Mapas en una imagen (columnas x filas) con su ficha de datos. `inicio` es el indice
    global del primer mapa (para numerarlos bien en paginas) y `extra` el texto del encabezado."""
    W, H, CAB = 4.4, 5.3, 1.1
    fig, filas, ancho, alto = _lienzo(len(mapas), columnas, W, H, CAB)
    if extra is None:
        viables = sum(r.viable for r in reportes)
        extra = f"viables: {viables}/{len(mapas)} ({100 * viables / len(mapas):.0f}%)"
    fig.text(0.5, 1 - 0.28 / alto, f"{titulo}   ·   {extra}", ha="center", fontsize=14, fontweight="bold")
    _leyenda(fig, 0.5, 1 - 0.55 / alto)
    for i, (m, r) in enumerate(zip(mapas, reportes)):
        f, c = divmod(i, columnas)
        x0, ytop = c * W, alto - CAB - f * H
        ax = fig.add_axes([(x0 + 0.4) / ancho, (ytop - 0.45 - 3.6) / alto, 3.6 / ancho, 3.6 / alto])
        dibujar_mapa(ax, m, r.ruta, r.viable)
        est, col = _estado(r)
        fig.text((x0 + 0.4) / ancho, (ytop - 0.12) / alto,
                 f"Mapa #{inicio + i + 1}  [{configs[i].nombre}]", fontsize=9, fontweight="bold", va="center")
        fig.text((x0 + 4.0) / ancho, (ytop - 0.12) / alto, est, fontsize=9, fontweight="bold",
                 color=col, ha="right", va="center")
        fig.text((x0 + 0.4) / ancho, (ytop - 0.30) / alto, _motivo(r), fontsize=6.5, color=ROJO, va="center")
        for k, linea in enumerate(_lineas(r)):
            fig.text((x0 + 0.4) / ancho, (ytop - 4.25 - k * 0.17) / alto, linea, fontsize=7, va="center")
    fig.savefig(ruta)
    plt.close(fig)


def guardar_comparacion(originales, rep_o, finales, rep_f, configs, nombre, pareo, ruta,
                        columnas=2, inicio=0, extra=None):
    """Cada celda: mapa original (izq) vs mapa tras el algoritmo (der), con sus datos."""
    W, H, CAB = 8.4, 5.3, 1.1
    fig, filas, ancho, alto = _lienzo(len(originales), columnas, W, H, CAB)
    n = len(originales)
    if extra is None:
        extra = f"viables {sum(r.viable for r in rep_o)}/{n} → {sum(r.viable for r in rep_f)}/{n}"
    if pareo == "ranking":
        extra += "  (AG: parejas por ranking de fitness)"
    fig.text(0.5, 1 - 0.28 / alto, f"{nombre}: original → resultado   ·   {extra}",
             ha="center", fontsize=14, fontweight="bold")
    _leyenda(fig, 0.5, 1 - 0.55 / alto)
    for i in range(n):
        f, c = divmod(i, columnas)
        x0, ytop = c * W, alto - CAB - f * H
        fig.text((x0 + 4.2) / ancho, (ytop - 0.12) / alto,
                 f"Mapa #{inicio + i + 1} [{configs[i].nombre}]   f(X): {_fmt(rep_o[i].fitness)} → {_fmt(rep_f[i].fitness)}",
                 fontsize=9, fontweight="bold", ha="center", va="center")
        for j, (mp, r, etiqueta) in enumerate(((originales[i], rep_o[i], "ORIGINAL"),
                                               (finales[i], rep_f[i], nombre))):
            xo = x0 + 0.3 + j * 4.0
            ax = fig.add_axes([xo / ancho, (ytop - 0.45 - 3.4) / alto, 3.4 / ancho, 3.4 / alto])
            dibujar_mapa(ax, mp, r.ruta, r.viable)
            est, col = _estado(r)
            fig.text((xo) / ancho, (ytop - 0.32) / alto, f"{etiqueta}: {est}",
                     fontsize=8, color=col, fontweight="bold", va="center")
            for k, linea in enumerate(_lineas(r)):
                fig.text(xo / ancho, (ytop - 4.05 - k * 0.17) / alto, linea, fontsize=6.4, va="center")
    fig.savefig(ruta)
    plt.close(fig)


def guardar_cuadricula_paginas(mapas, reportes, configs, titulo, carpeta, prefijo,
                               por_pagina=12, columnas=3):
    """Divide la poblacion en paginas legibles (por defecto 12 mapas = 3x4). Devuelve las rutas."""
    n = len(mapas)
    viables = sum(r.viable for r in reportes)
    extra = f"viables en toda la población: {viables}/{n} ({100 * viables / n:.0f}%)"
    paginas = math.ceil(n / por_pagina)
    rutas = []
    for p in range(paginas):
        a, b = p * por_pagina, min(n, (p + 1) * por_pagina)
        ruta = carpeta / f"{prefijo}_p{p + 1:02d}.png"
        guardar_cuadricula(mapas[a:b], reportes[a:b], configs[a:b],
                           f"{titulo} · mapas {a + 1}-{b} (pág. {p + 1}/{paginas})", ruta,
                           columnas, inicio=a, extra=extra)
        rutas.append(ruta)
    return rutas


def guardar_comparacion_paginas(originales, rep_o, finales, rep_f, configs, nombre, pareo,
                                carpeta, prefijo, por_pagina=6, columnas=2):
    """Igual que guardar_cuadricula_paginas pero con parejas original -> resultado (6 por pagina)."""
    n = len(originales)
    extra = (f"viables en toda la población: {sum(r.viable for r in rep_o)}/{n} → "
             f"{sum(r.viable for r in rep_f)}/{n}")
    paginas = math.ceil(n / por_pagina)
    rutas = []
    for p in range(paginas):
        a, b = p * por_pagina, min(n, (p + 1) * por_pagina)
        ruta = carpeta / f"{prefijo}_p{p + 1:02d}.png"
        guardar_comparacion(originales[a:b], rep_o[a:b], finales[a:b], rep_f[a:b], configs[a:b],
                            nombre, pareo, ruta, columnas, inicio=a,
                            extra=f"{extra}  ·  pág. {p + 1}/{paginas}")
        rutas.append(ruta)
    return rutas


def guardar_galeria(paginas, titulo, ruta_html):
    """Un index.html que apila todas las paginas (scroll + zoom del navegador)."""
    n = len(paginas)
    nav = " · ".join(f'<a href="#p{i}">{i}</a>' for i in range(1, n + 1))
    cuerpo = "\n".join(f'<h3 id="p{i}">Página {i}/{n}</h3><img src="{p.name}" loading="lazy">'
                       for i, p in enumerate(paginas, 1))
    html = ("<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>{titulo}</title><style>body{{font-family:sans-serif;background:#2b2b2b;"
            "color:#eee;margin:0;padding:16px}}img{{max-width:100%;height:auto;display:block;"
            "margin:0 auto 32px;background:#fff}}h1,h3,p{{text-align:center}}a{{color:#8ab4f8}}"
            f"</style></head><body><h1>{titulo}</h1><p>{nav}</p>{cuerpo}</body></html>")
    ruta_html.write_text(html, encoding="utf-8")


def guardar_convergencia(resultados, ruta, titulo="Convergencia del fitness"):
    """resultados: dict nombre -> Resultado (con .mejor y .media por paso)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for nombre, r in resultados.items():
        x = np.arange(len(r.mejor))
        axes[0].plot(x, r.mejor, label=nombre, color=COLOR_ALGO[nombre], linewidth=2)
        axes[1].plot(x, r.media, label=nombre, color=COLOR_ALGO[nombre], linewidth=2)
    for ax, t in zip(axes, ("Mejor fitness de la población", "Fitness medio de la población")):
        ax.set_yscale("log")
        ax.set_xlabel("Paso (generación / iteración / paso de recocido)")
        ax.set_ylabel("f(X)  (menor es mejor, escala log)")
        ax.set_title(t)
        ax.grid(alpha=0.3, which="both")
        ax.legend()
    fig.suptitle(titulo, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(ruta, dpi=130)
    plt.close(fig)


def guardar_resumen(stats, fitness_por_algoritmo, ruta, titulo="Resumen comparativo"):
    """stats: lista de dicts (incluye 'Inicial'); fitness_por_algoritmo: nombre -> lista de f."""
    nombres = [s["algoritmo"] for s in stats]
    colores = [COLOR_ALGO[n] for n in nombres]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    ax = axes[0, 0]
    bp = ax.boxplot([fitness_por_algoritmo[n] for n in nombres], tick_labels=nombres, patch_artist=True)
    for patch, col in zip(bp["boxes"], colores):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)
    ax.set_yscale("log")
    ax.set_title("Distribución del fitness final (menor = mejor)")

    def barras(ax, clave, titulo_, fmt="{:.1f}"):
        vals = [s[clave] for s in stats]
        b = ax.bar(nombres, vals, color=colores)
        ax.set_title(titulo_)
        for rect, v in zip(b, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(), fmt.format(v),
                    ha="center", va="bottom", fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    barras(axes[0, 1], "pct_viables", "% de mapas viables", "{:.0f}%")
    barras(axes[0, 2], "error_dificultad", "Error medio de dificultad |D − objetivo|")
    barras(axes[1, 0], "diversidad", "Diversidad (Hamming medio entre mapas)")
    barras(axes[1, 1], "desv_fitness", "Variabilidad del fitness (desv. estándar, escala log)")
    axes[1, 1].set_yscale("log")
    ax = axes[1, 2]
    algos = [s for s in stats if s["algoritmo"] != "Inicial"]
    ax.bar([s["algoritmo"] for s in algos], [s["tiempo"] for s in algos],
           color=[COLOR_ALGO[s["algoritmo"]] for s in algos])
    for i, s in enumerate(algos):
        ax.text(i, s["tiempo"], f"{s['tiempo']:.1f}s", ha="center", va="bottom", fontsize=9)
    ax.set_title("Tiempo de cómputo (s)")
    ax.grid(axis="y", alpha=0.3)
    fig.suptitle(titulo, fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    plt.close(fig)


def guardar_mejores_mapas(mejores, ruta, titulo="Mejor mapa de cada algoritmo"):
    """mejores: dict nombre -> (mapa, Reporte)."""
    n = len(mejores)
    fig = plt.figure(figsize=(5.6 * n, 7.4))
    fig.suptitle(titulo, fontsize=15, fontweight="bold", y=0.98)
    _leyenda(fig, 0.5, 0.945)
    for k, (nombre, (mapa, r)) in enumerate(mejores.items()):
        ax = fig.add_axes([(k * 5.6 + 0.4) / (5.6 * n), 0.25, 4.8 / (5.6 * n), 4.8 / 7.4 * 0.95])
        dibujar_mapa(ax, mapa, r.ruta, r.viable)
        est, col = _estado(r)
        ax.set_title(f"{nombre} — {est}", color=col, fontsize=12, fontweight="bold")
        for i, linea in enumerate(_lineas(r)):
            fig.text((k * 5.6 + 0.4) / (5.6 * n), 0.2 - i * 0.027, linea, fontsize=8.5)
    fig.savefig(ruta, dpi=110)
    plt.close(fig)