import argparse

from generator.map_utils import fancy_print, generate_random_map
from generator.scenarios import SCENARIOS

def main() -> None:
    #Deja seleccionar el escenarios desde la termnal para generar el mapa con los rangos que corresponden
    #EJEMPLO: uv run map --escenario facil
    parser = argparse.ArgumentParser(description="Generador de mapas")
    parser.add_argument(
        "--escenario", choices=SCENARIOS, default="medio",
        help="Tipo de mapa que se generara (default: medio)",
    )
    args = parser.parse_args()

    print ("Escenario seleccionado:", args.escenario)
    fancy_print(generate_random_map(args.escenario))