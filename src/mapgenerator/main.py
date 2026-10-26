#Oswaldo tambien hizo esto, pero ya ni lo reviso alv, es pura ia

import os
import json
from generator.map_utils import generar_y_guardar_mapas
from generator.genetic import Genetic

def main():
    print("==================================================")
    print(" INICIANDO EL PROYECTO SUPREMO DE MAPAS DE OSWALDO")
    print("==================================================\n")
    
    # 1. FASE DE CREACIÓN GLOBAL
    print("[FASE 1] Forjando el universo base...")
    generar_y_guardar_mapas(cantidad=100)
    
    # 2. LECTURA DEL JSON
    # Navegamos un nivel arriba de mapgenerator para llegar a la raíz (SIC) y entrar a Mapas
    ruta_actual = os.path.dirname(os.path.abspath(__file__))
    ruta_raiz = os.path.dirname(ruta_actual)
    ruta_json = os.path.join(ruta_raiz, "Mapas", "poblacion_inicial.json")
    
    with open(ruta_json, "r", encoding="utf-8") as archivo:
        mapas_universales = json.load(archivo)
        
    print(f"[FASE 2] Universo cargado exitosamente. {len(mapas_universales)} mapas en la recámara.\n")
    
    # 3. FASE DE EVOLUCIÓN (ALGORITMO GENÉTICO)
    print("==================================================")
    print(" ARRANCANDO ALGORITMO GENÉTICO")
    print("==================================================")
    
    # Instanciamos a Oswaldo, y le pasamos los mapas universales en la mano para que no haga trampa
    ag = Genetic(poblacion_base=mapas_universales)
    
    # Lo ponemos a entrenar por 20 generaciones (puedes subirlo a 50 o 100 después)
    # la primera generacion son los mapas crudos sacados del json, no se de donde los saco json, ni porque 
    # trae tantos mapas, YO NUNCA ESTUVE DE ACUERDO EN METERLO AL EQUIPO
    mejor_mapa_genetico, historia_genetico = ag.entrenar(total_generaciones=100)
    
    print("\n==================================================")
    print(" RESULTADOS FINALES")
    print("==================================================")
    print(f"El Algoritmo Genético terminó con un Mejor Fitness de: {mejor_mapa_genetico['fitness']}")

if __name__ == "__main__":
    main()  

#Aqui oswaldo ya se fue a dormir, mañana con mas calma revisa que pedo, porque aunque quiere seguir, lamentablemente 
# debe dormir   