from typing import Annotated, NotRequired, TypedDict

from representation import Elemento

Rangos = dict[Elemento, tuple[int, int]]

DEFAULT_CONSTRAINTS: Annotated[Rangos, "Rango de objetos posibles dentro del mapa."] = {
    Elemento.OBSTACULO: (20, 35),
    Elemento.ENEMIGO: (5, 10),
    Elemento.RECOMPENZA: (3, 6),
}


class ScenarioConfig(TypedDict):
    dificultad: int | None
    rangos: Rangos
    camino_largo: bool


class _ScenarioDef(TypedDict):
    dificultad: NotRequired[int]
    rangos: NotRequired[Rangos]
    camino_largo: NotRequired[bool]


SCENARIOS: dict[str, _ScenarioDef] = {
    "facil": {"dificultad": 30},
    "medio": {"dificultad": 50},
    "dificil": {"dificultad": 70},
    "laberinto": {"rangos": {Elemento.OBSTACULO: (35, 50)}},
    "combate": {"rangos": {Elemento.ENEMIGO: (10, 15)}},
    "exploracion": {
        "rangos": {Elemento.RECOMPENZA: (6, 10)},
        "camino_largo": True,
    },
}


def get_scenario(nombre: str | None) -> ScenarioConfig:
    """
    Obtiene la configuración y restricciones del escenario especificado.

    Args:
        nombre: Nombre del escenario a consultar (facil, medio, dificil,
            laberinto, combate o exploracion). Si es None se devuelve la
            configuración predeterminada.

    Returns:
        ScenarioConfig: Diccionario con la dificultad objetivo, los rangos de
        elementos válidos y si requiere camino largo.

    Raises:
        ValueError: Si el nombre del escenario no existe en SCENARIOS.
    """
    if nombre is None:
        config: _ScenarioDef = {}
    else:
        nombre = nombre.strip().lower()
        if nombre not in SCENARIOS:
            raise ValueError(
                f"Escenario desconocido: '{nombre}'. "
                + f"Opciones disponibles: {', '.join(SCENARIOS)}"
            )
        config = SCENARIOS[nombre]

    rangos: Rangos = DEFAULT_CONSTRAINTS.copy()
    rangos.update(config.get("rangos", {}))

    return {
        "dificultad": config.get("dificultad"),
        "rangos": rangos,
        "camino_largo": config.get("camino_largo", False),
    }
