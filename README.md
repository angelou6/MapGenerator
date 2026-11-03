# Mapa Procedural

Hola chavos :D!!!

## Setup

Van a ocupar instalar [Uv](https://docs.astral.sh/uv/getting-started/installation/) para correr el proyecto.

Utilizan este comando para correr la funcion `main` dentro de [mapgenerator/main.py](src/mapgenerator/main.py)

```
$ uv run map
```

## Uso

```
$ uv run map                 # flujo interactivo (población -> preset -> AG -> PSO -> RS -> resultados)
$ uv run map --lote          # 180 corridas (3 algoritmos x 6 escenarios x 10) + tablas y gráficas
$ uv run map --lote --procesos 4   # igual, en paralelo
```

Opciones útiles: `--pesos w1 w2 w3 w4 w5`, `--presupuesto N` (generaciones = iteraciones = pasos
de recocido por mapa, default 150), `--semilla N` (repetir una ejecución), `--poblacion N` y
`--preset NOMBRE` (saltan las preguntas), `--sin-pausas`, `--no-abrir`, `--salida CARPETA`.

Las imágenes y tablas se guardan en `resultados/sesion_<fecha>/` o `resultados/lote_<fecha>/`.

### Definiciones usadas

- **Mapa viable**: hay camino inicio-meta, las cantidades están en rango, ninguna recompensa
  encerrada, ningún enemigo aglomerado (P_enemigos = 0) y |D - D_objetivo| <= 10
  (`TOLERANCIA_DIFICULTAD` en `generator/evaluacion.py`). Exploración también exige L >= 40.
- **Presupuesto justo**: AG, PSO y RS reciben la misma población inicial y ~el mismo número de
  evaluaciones por mapa.
- **Ganador**: menor posición promedio en fitness medio, % viables, error de dificultad,
  diversidad (Hamming), estabilidad (desv. del fitness) y tiempo.

## Recursos

- https://youtu.be/kHyNqSnzP8Y
- https://youtu.be/vhSBqk6SAB4
- https://youtu.be/v_-xfULBu2I
- https://youtu.be/Xe63l4kpuMg
- https://youtu.be/21EDdFVMz8I
