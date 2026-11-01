#configuracion de los escenarios facil, dificil, medio, combate, exploracion y laberinto
from generator.map_utils import CONSTRAINTS
from representation import Elemento

#Guarda los escenarios en un solo lugar
#Facial, medio y dificil usan los rangos normales y cambian la dificultad objetivo
#los otros cambian solo los rangos que hay en el doc
SCENARIOS = {
    "facil": {"dificultad": 30},
    "medio": {"dificultad": 50},
    "dificil": {"dificultad": 70},
    "laberinto": {"rangos": {Elemento.OBSTACULO: (35, 50)}},
    "combate": {"rangos": {Elemento.ENEMIGO: (10,15)}},
    "exploracion": {
        "rangos": {Elemento.RECOMPENZA: (6,10)},
        "camino_largo": True,
        },
}

def get_scenario(nombre: str) -> dict:
    #Devolvera la dificultad, los rangos y las condiciones de camino del escenario indicado
    nombre = nombre.strip().lower()
    if nombre not in SCENARIOS:
        raise ValueError(f"Escenario desconocido: {nombre}. Opciones: {', '.join(SCENARIOS)}")

    config = SCENARIOS[nombre]
    #parte de los rangos normales y sustituir solo los que cambian en en el escenario
    rangos = CONSTRAINTS.copy()
    rangos.update(config.get("rangos", {}))

    #no hay dificultad ni longitud definida para los escenarios especiales asi que XD
    return {
        "dificultad": config.get("dificultad"),
        "rangos": rangos,
        "camino_largo": config.get("camino_largo", False),
    }