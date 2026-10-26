#Aqui empieza el jale de oswaldo
"""
Oswaldo hizo esto, oswaldo intentaba hacerlo bien mientras su mente pensaba en comida... no deberias confiar en 
oswaldo mientras piensa en comida
"""
# Oswaldo importo las funciones que ya habian hecho para usarlas con los nuevos mapas y luego importo el random
from generator.map_utils import (
    generate_random_map,
    calculate_distribution_penalty,
    calculate_enemy_penalty,
    calculate_path_penalty,
    calculate_reward_penalty,
    get_difficulty
)
import random

# Oswaldo importo la funcion de busqueda de camino para poder calcular la dificultad del mapa
from path_search.breadth import bfs

#Oswaldo importa tambien numpy, para poder hacer la mezcla de los mapas, oswaldo vio que lo usaba otro script
#del proyecto, asi que decidio usarlo tambien aqui
import numpy as np

class Genetic:
    def __init__(self, population_size=100, limites_manuales=None, poblacion_base=None):
            """
            Oswaldo, el genio de la genética de mapas, ha decidido crear una clase para manejar la evolución de mapas.
            :param population_size: es el tamaño de la poblacion de mapas de cada generacion,
            oswaldo piensa que 100 es un buen numero.
            :param limites_manuales: Diccionario opcional por si el usuario se pone exigente y quiere usar sus propios rangos.
            """
            self.population_size = population_size
            self.population = []  # oswaldo guardo la lista de mapas aqui, oswaldo dice que esa es la pobracion inicial
            self.generation = 1   # este es un contador de generaciones, oswealdo piensa que suena estupido, pero quien es
            #el para para decir lo que es estupido y lo que no
            
            # Si Oswaldo recibe un cargamento de mapas desde el main (el json), los procesa aqui
            if poblacion_base is not None:
                for mapa in poblacion_base:
                    # El JSON guarda listas normales (ta bien wey), pero el evaluador pide Numpy Arrays
                    matriz_np = np.array(mapa["matrix"])
                    self.population.append({"matrix": matriz_np, "fitness": None})
                self.population_size = len(poblacion_base)
            
            # Oswaldo tiene que saber si los limites que le envieron estan puestos por el usuario, o si debe usar los que
            # el manual le indica
            if limites_manuales is None:
                self.limites = {
                    1: (20, 35),  # Obstáculos (1)
                    2: (5, 12),   # Enemigos (2)
                    3: (3, 8)     # Recompensas (3)
                }
            else:
                self.limites = limites_manuales

    def generate_initial_population(self):
        """
        Oswaldo ha decidido crear la poblacion inicial aqui, son completamente aleatorios
        """
        self.population = []
        
        for _ in range(self.population_size):
            # 1. Oswaldo decidio generar un mapa aleatorio usando la que ya tenemos
            nuevo_mapa = generate_random_map()
            
            # 2. Oswaldo empaqueta el mapa en un diccionario, este mapa sera un individuo
            individuo = {
                "matrix": nuevo_mapa,
                "fitness": None  # Oswaldo ya sabe cómo calcularlo, ocurre en la función de abajo
            }
            
            # 3. Ahora oswaldo agrega el individuo a la poblacion
            self.population.append(individuo)
            
        print(f"Población inicial generada: {len(self.population)} mapas listos.")
        return self.population

    def evaluar_poblacion(self):
        """
        Oswaldo agarro el pedo luego de comer e hizo esto para evaluar que tan buenos son los mapas 
        Fitness = Dificultad base - Penalizaciones.
        """
        for individuo in self.population:
            matriz_del_mapa = individuo["matrix"]
            
            # 1. Oswaldo manda a buscar el camino real con su confiable BFS (en realidad no confio en el, se porta
            # raro a veces, hace rato lo vio comiendo un cheto con salsa verde, quien hace eso?)
            resultado_bfs = bfs(matriz_del_mapa)
            
            # 2. Oswaldo decidio calcular todas las penalizaciones aqui, pasandole el resultado de la busqueda
            penalidad_distribucion = calculate_distribution_penalty(matriz_del_mapa)
            penalidad_enemigos = calculate_enemy_penalty(matriz_del_mapa)
            penalidad_caminos = calculate_path_penalty(matriz_del_mapa, solution=resultado_bfs)
            penalidad_recompensas = calculate_reward_penalty(matriz_del_mapa)
            
            suma_total_penalizaciones = (
                penalidad_distribucion + 
                penalidad_enemigos + 
                penalidad_caminos + 
                penalidad_recompensas
            )
            
            # 3. Oswaldo calcula la dificultad del mapa en bruto usando la ruta de verdad
            dificultad_base = (
                get_difficulty(matriz_del_mapa, solution=resultado_bfs)
                if resultado_bfs is not None
                else 0
            )
            
            # 4. Oswaldo calcula el fitness final (Oswaldo multiplica por 10 para que si el mapa es muy malo, 
            # lo mande pero a chingar a su madre)
            calificacion_final = dificultad_base - (suma_total_penalizaciones * 10)
            
            # 5. Oswaldo guarda la calificacion final en el individuo, y les recuerda que el individuo es un mapa de la 
            # poblacion, por eso es un individuo, jeje
            individuo["fitness"] = calificacion_final

    def torneo_seleccion(self, tamaño_torneo=3):
        """
        Oswaldo se embriago de poder, y ahora hizo un club de la pelea para los mapas.
        elige el tamaño de torno y participantes al azar y sobrevive el mejor

        primera regla del club de la pelea: siempre habla del club de la pelea

        """
        # Oswaldo selecciona los mapas participantes al azar para el torneo, aunque puedes entrar 
        # mas facilmente si tienes una hamburguesa con su nombre
        participantes = random.sample(self.population, tamaño_torneo)
        
        # Oswaldo no decide quien gana, el mejor mapa es el que gana, lo siento bobos, aqui no hay 
        #anvorguesas que los puedan ayudar
        padre_ganador = max(participantes, key=lambda individuo: individuo["fitness"])
        
        return padre_ganador

    def cruzar_padres(self, padre1, padre2):
        """
        Oswaldo ya esta hasta la madre de jalar, pero no puede dejar de hacerlo porque ya se hizo wey toda la semana...
        despues de todo oswaldo tiene verguenza, algo asi

        Oswaldo decide que es hora de que los padres se tomen de la mano y tengan un hijo, 
        aunque no sabe si es buena idea, pero bueno, el amor es ciego y oswaldo tambien

        Para esto tomaremos regiones de 5x5 del mapa de cada padre y las colocaremos para mezclar un nuevo mapa hijo
        no es la manera mas bonita de hacerlo, pero estamos hablando de algoritmos, esperaban mas? (yo si)

        """
        # 1. Oswaldo tomara primero una copia del mapa papa y luego tomamos el mapa mama
        # no se preocupen, el hijo no es un clon, es un nuevo mapa con genes de ambos papis
        mapa_hijo = np.copy(padre1["matrix"])
        mapa_padre2 = padre2["matrix"]
        
        # 2. Oswaldo restringue las regiones 0 y 8 para que no se mezclen, porque son las esquinas y esas no
        # las queremos mezclar
        regiones_intercambiables = [1, 2, 3, 4, 5, 6, 7]
        
        # 3. Oswaldo es incapaz de decidir cuantos cuadros interambiaran, ademas de que no podemos preguntarle siempre.
        # Asi que decidio poner un random que lo decidira por el. Entre 2 y 4 regiones suenan bien para el 
        cantidad_a_intercambiar = random.randint(2, 4)
        
        # 4. Oswaldo ve que regiones son las que su algoritmo decidio seleccionar e intercambiar
        regiones_elegidas = random.sample(regiones_intercambiables, cantidad_a_intercambiar)
        
        # 5. LOswaldo manda a los papis al quirófano y les saca las regiones elegidas para ponerlas en el hijo
        #  No se preocupen, los papis estan bien
        for region in regiones_elegidas:
            # Hola, soy el cirujano encargado de esta operacion, oswaldo me dijo que les explique que haremos asi que 
            # por favor acompañenme
            # Imaginemos un gato, donde cada casilla es una region de papi o mami, el gato tiene 3x3 regiones, y cada 
            # region se divide entre 3, si lo multiplicamos por 5, obtenemos la fila  inicial de cada region
            fila_inicio = (region // 3) * 5
            fila_fin = fila_inicio + 5
            
            # aqui hacemos lo mismo para la columna, pero usando el modulo... modulo?
            # no se, honestamente no termine la especialidad... no creo que usemos estos terminos en mi 
            # especialidad, pero aqui lo hacemos para obtener la columna inicial 
            col_inicio = (region % 3) * 5
            col_fin = col_inicio + 5
            
            # Ahora hacemos un transplante de la region de la mami al papi, y el hijo es el resultado de este
            #  bello procedimiento... Adios :D
            mapa_hijo[fila_inicio:fila_fin, col_inicio:col_fin] = mapa_padre2[fila_inicio:fila_fin, col_inicio:col_fin]
            
        # 6. Oswaldo ahora saca al nuevo engendro y lo guarda en un lugar magico llamado diccionario, 
        # donde la dara un numero para identificarlo y su fitness, el cual es none porque acaba de nacer
        # JAJAJAJAJA QUE WEY, NACIO EN 2026 XD
        hijo = {
            "matrix": mapa_hijo,
            "fitness": None
        }
        
        return hijo

    def mutar_hijo(self, hijo, probabilidad_mutacion=0.10):
        """
        Oswaldo ahora necesita saber si el hijo es fuerte para sobrevivir, pero, QUE, SUS PAPAS ERAN PRIMOS?!.
        Inaudito, que terrible suceso, pero bueno, ya no podemos cambiar el pasado, solo arreglar el presente si
        esperamos tener un mejor futuro....

        Oswaldo deberia escucharse a si mismo de vez en cuando

        Bueno, respecto al hijo, le haremos unas pruebas para mutar algunos de sus genes y volverlo un mapa fuerte
        que cumpla con los estandares de mapas
        """
        # 1. Oswaldo sabe que segun los estudios, los algoritmos mutan entre un 5 y 10 por ciento de sus genes, asi que
        # decidio poner un 10% de probabilidad de mutacion
        if random.random() < probabilidad_mutacion:
            matriz = hijo["matrix"]
            filas = len(matriz)
            columnas = len(matriz[0])
            
            while True:
                # 2. Oswaldo ya sabe que hay que mutar una casilla al azar, asi que elegimos una fila y columna 
                f_azar = random.randint(0, filas - 1)
                c_azar = random.randint(0, columnas - 1)
                
                # 3. PERO ESPERA, Oswaldo se dio cuenta que no podemos mutar la casilla de inicio ni la de meta, 
                # asi que las excluimos
                es_inicio = (f_azar == 0 and c_azar == 0)
                es_meta = (f_azar == filas - 1 and c_azar == columnas - 1)
                
                if not es_inicio and not es_meta:
                    # 4. Ahora elegiremos un numero al azar del 0 al 4 y luego de eso,
                    # Oswaldo se asegura de que sea diferente al que ya tenía
                    valor_actual = matriz[f_azar][c_azar]
                    opciones = [0, 1, 2, 3, 4]
                    if valor_actual in opciones:
                        opciones.remove(valor_actual)
                    
                    nuevo_valor = random.choice(opciones)
                    matriz[f_azar][c_azar] = nuevo_valor
                    break # Una vez que eso pase, dejamos en paz al individuo y salimos del bucle
                    
        return hijo

    def reparar_hijo(self, hijo):
        """
        Oswaldo ahora piensa que el hijo recien nacido es fuerte, pero arreglar algo asi no es tan sencillo,
        ya le aplicamos el tratamiento, hace falta ver que haya quedado bien
        """
        matriz = hijo["matrix"]
        filas = len(matriz)
        columnas = len(matriz[0])
        
        # 1. Ahora Oswaldo va a revisar que el niño este entre el rango estandar de mapas (o los manuales)
        # Jarbis le recuerda a Oswaldo que los limites ahora se jalan desde self.limites guardados en el __init__
        
        # 2. Oswaldo saca su libreta para hacer un conteo de cuantos genes tiene de cada uno... oh si, lo estoy viendo...
        # sisi, efectivamente, 2 de estos, sisi
        posiciones = {0: [], 1: [], 2: [], 3: [], 4: []}
        
        for f in range(filas):
            for c in range(columnas):
                # Oswaldo ha terminado de hacer inventario en este punto, y ahora toca ponerse la armadura para defender
                # la ciudad santa de inicio y meta, Oswaldo esta listo para tomar Jerusalen si es necesario
                if (f == 0 and c == 0) or (f == filas - 1 and c == columnas - 1):
                    continue
                    
                valor = matriz[f][c]
                if valor in posiciones:
                    posiciones[valor].append((f, c))
                    
        # 3. Oswaldo perdio una pierna y 2 dientes defendiendo la tierra sagrada, pero ahora que esta a salvo
        #  iniciamos la reparación para cada elemento, a continuacion, Oswaldo nos explicara que pasara en cada caso
        for tipo, (minimo, maximo) in self.limites.items():
            cantidad_actual = len(posiciones[tipo])

            # ola, soy oswaldo, y esta es mi primera vez hablando, bueno no, si hablaba yo desde antes pero esta es mi 
            # primera vez hablando aqui en el proyecto, y bueno, veamos los casos
            
            # En el caso a: hay demasiado elementos, asi que les metemos lumbre y los hacemos piso
            if cantidad_actual > maximo:
                exceso = cantidad_actual - maximo
                a_borrar = random.sample(posiciones[tipo], exceso)
                
                for f, c in a_borrar:
                    matriz[f][c] = 0
                    posiciones[0].append((f, c))     # Y pues aqui ya es espacio libre
                    posiciones[tipo].remove((f, c))  # Y aqui nos encargamos de deshacernos del elemento
                    
            # En el caso b: hay muy pocos elementos, asi que les voy a poner cosas a algunas casillas que sean piso, como
            # a esta, le voy a pegar... este chicle... es que ya no tiene sabor
            elif cantidad_actual < minimo:
                faltante = minimo - cantidad_actual
                
                # Ahora solo revisamos que haya suficientes espacios para rellenar, mas vale prevenir, no vaiga a ser
                if len(posiciones[0]) >= faltante:
                    a_rellenar = random.sample(posiciones[0], faltante)
                    
                    for f, c in a_rellenar:
                        matriz[f][c] = tipo
                        posiciones[tipo].append((f, c))  # Y aqui pues le ponemos el elemento
                        posiciones[0].remove((f, c))     # Y aqui ya no es espacio libre

                        # y bueno, eso es mi explicacion amigos, ahora volvamos con el narrador...
                        #...
                        #...
                        #...
                        #... Jarbis..?
                        #... no me abandones...
                        
        # Oswaldo ha revisado al nuevo miembro de la poblacion de mapas, y resulta que todo esta bien, es un individuo
        # sano y fuerte... 3 hurras por el nuevo
        # HIP HIP... HURRA!
        # HIP HIP... HURRA!
        # HIP HIP... HURRA!
        hijo["matrix"] = matriz
        return hijo

    def generar_siguiente_poblacion(self):
        """
        Oswaldo esta emocionado, ha llegado una de las mejores partes de este proyecto, y no es para menos muchachos
        es hora de implementar...

        RACISMO!!!!!

        Asi es muchachos, aplicaremos racismo, esto nos permitira elegir solamente a los individuos mas aptos para 
        nuestra nueva y mas refinada poblacion de mapas

        CREAREMOS UNA GENERACION DE MAPAS SUPERIOR, DONDE SOLO LOS MAS PUROS Y FUERTES LLEGUEN A DONDE LOS DEMAS
        NO PUEDAN LLEGAR.

        ICH HABE NOCH MEHR HUNGER VERDAMMT! ES KANN DOCH NICHT SEIN, DASS ICH IMMER HUNGER HABE

        HAIL HIhahab...

        Lamentamos los imprevistos, continuemos con el reporte
        """
        nueva_poblacion = []
        
        # 1. RACISMO, con esta herramienta salvaremos a nuestra creacion de la mediocridad
        mejor_individuo = max(self.population, key=lambda individuo: individuo["fitness"])
        
        # clonamos el mapa para que no contamine a las nuevas generaciones con sus parametros
        mapa_elite = {
            "matrix": np.copy(mejor_individuo["matrix"]),
            "fitness": mejor_individuo["fitness"]
        }
        nueva_poblacion.append(mapa_elite)
        
        # 2. Aqui oswaldo rellena el resto de la cuota generacional, menos al elite... wort wort wort
        for _ in range(self.population_size - 1):
            padre_1 = self.torneo_seleccion()
            padre_2 = self.torneo_seleccion()
            
            hijo = self.cruzar_padres(padre_1, padre_2)
            hijo_mutante = self.mutar_hijo(hijo)
            hijo_legal = self.reparar_hijo(hijo_mutante)
            
            nueva_poblacion.append(hijo_legal)
            
        # Oswaldo comete un genocidio contra toda la poblacion que no paso las pruebas, convenientemente casi 
        # todos tenian un exceso de melanina jaja, que curioso (ojitos asi bien void)
        self.population = nueva_poblacion
        self.generation += 1
        
        print(f"¡La generacion {self.generation} ha nacido! {len(self.population)} mapas (1 Élite y {self.population_size - 1} mutantes).")

    def entrenar(self, total_generaciones=50):
        """
        Oswaldo ahora necesita ver como estas generaciones evolucionan al paso de los siglos, 
        pero no podemos esperar tanto, tenemos que dormir, ya es talde, y mañana tlabaja, tiene
        que tlabajal
        """
        self.historial = [] # Oswaldo prepara su cuaderno para notar el milagro de la evolucion
        
        # 1. El Genesis, adan y eva junto a sus 98 hijos son parte de la poblacion inicial
        print("--- DIA 1: EL GENESIS ---")
        self.generate_initial_population()
        self.evaluar_poblacion()
        
        # 2. Oswaldo no espera a que su maruchan este lista, no piensa esperar a que hagan 50 generaciones
        # asi que aceleremos esto
        for epoca in range(1, total_generaciones + 1):
            
            # Oswaldo les pasa lista y los pone en escala de niveles de poder con una cancion de linkin park de fondo
            # oswaldo tiene un gusto exquisito si me lo preguntan
            fitness_lista = [ind["fitness"] for ind in self.population]
            mejor = max(fitness_lista)
            peor = min(fitness_lista)
            promedio = sum(fitness_lista) / len(fitness_lista)
            
            # Oswaldo ya esta pensando en el futuro, asi que hacemos guardado de datos para las graficas cuando 
            # el experimento termine
            registro = {
                "generacion": self.generation,
                "mejor": mejor,
                "peor": peor,
                "promedio": promedio
            }
            self.historial.append(registro)
            
            # Oswald0, podrias mostrarnos ese reporte que tienes ahi?... ese, mmm si, ahi tienes todas las estadisticas
            # y los datos... y aqui esta la media
            print(f"Época {self.generation} | Mejor Fitness: {mejor} | Peor: {peor} | Promedio: {promedio:.2f}")
            
            # Oswaldo, ahora necesitamos que revises si ya no hay mas generaciones por las cuales pasar...
            if epoca < total_generaciones:
                self.generar_siguiente_poblacion()
                self.evaluar_poblacion() # Y nunca olvides evaluar a los recien nacidos
                
        # 3. Al final de los tiempos, Oswaldo se da a la tarea de buscar al mas perron aqui
        mejor_historico = max(self.population, key=lambda x: x["fitness"])
        print("\nSimulacion terminada")
        print(f"El mapa mapa optimo de las generaciones tiene un fitness de: {mejor_historico['fitness']}")
        
        # Lo ponemos como el mas cabron de todos los tiempos, y lo regresamos 
        return mejor_historico, self.historial

#Aqui termina el trabajo de oswaldo, por ahora, oswaldo consiguio comida mientras terminaba esto, pero no hubo coca
