import numpy as np

class ParticleSwarm:
    def __init__(self):
        """
        Oswaldo es un hombre de muchas facetas, despues de su desliz con las generaciones de mapas, decidimos
        pasarlo a un campo mas tranquilo, ahora es el señor de las abejas
        """
        pass

    def traducir_particula_a_mapa(self, matriz_flotante):
        """
        Las abejas de oswaldo vienen de decimalandia, por lo cual, Oswaldo no puede entenderlas, el no habla
        taka taka, pero en su intento de hacer un buen trabajo, Oswaldo saca su diccionario decimal-entero 
        para hacer la traducccion, y que el evaluador no explote
        """
        filas = len(matriz_flotante)
        columnas = len(matriz_flotante[0])
        
        # En un gito inesperado de las cosas, oswaldo toma la pala, y 
        mapa_fisico = np.zeros((filas, columnas), dtype=int)
        
        for f in range(filas):
            for c in range(columnas):
                # 1. Oswaldo protege la entrada (0,0) y la salida (14,14) con su vida. Siempre son piso (0).
                if (f == 0 and c == 0) or (f == filas - 1 and c == columnas - 1):
                    mapa_fisico[f][c] = 0
                    continue
                
                valor = matriz_flotante[f][c]
                
                # 2. Diccionario de traducción de Oswaldo (basado estrictamente en la Tarea #25):
                
                # Menor a 0.70 -> Camino libre (0)
                if valor < 0.70:
                    mapa_fisico[f][c] = 0
                
                # De 0.70 a 0.84 -> Obstáculo (1)
                elif 0.70 <= valor <= 0.84:
                    mapa_fisico[f][c] = 1
                
                # De 0.85 a 0.94 -> Enemigo (2)
                elif 0.85 <= valor <= 0.94:
                    mapa_fisico[f][c] = 2
                
                # De 0.95 en adelante -> Recompensa (3)
                else:
                    mapa_fisico[f][c] = 3
                    
        return mapa_fisico