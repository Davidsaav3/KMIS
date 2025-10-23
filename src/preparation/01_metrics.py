import pandas as pd  # MANEJO DE DATAFRAMES
import os            # MANEJO DE RUTAS Y DIRECTORIOS

# PARÁMETROS 
INPUT_FILE = '../../data/dataset.csv'                # RUTA DEL DATASET DE ENTRADA
OUTPUT_FILE = '../../results/preparation/01_metrics.csv'  # RUTA DEL CSV DE SALIDA
INCLUDE_DESCRIBE = 'all'                             # TIPO DE COLUMNAS PARA DESCRIBE: 'all', 'number', 'object'
SHOW_INFO = True                                     # TRUE = MOSTRAR MENSAJES EN PANTALLA

# CARGAR DATASET
df = pd.read_csv(INPUT_FILE)                         # LEER CSV A DATAFRAME
if SHOW_INFO:
    print(f"[ INFO ] Dataset cargado: {df.shape[0]} FILAS, {df.shape[1]} COLUMNAS")

# CALCULAR ESTADÍSTICAS DESCRIPTIVAS
info = df.describe(include=INCLUDE_DESCRIBE).transpose()  # DESCRIPCIÓN CON TRASNPOSICIÓN #
info['nulos'] = df.isnull().sum()                         # CONTAR VALORES NULOS POR COLUMNA
if SHOW_INFO:
    print("[ INFO ] Estadísticas descriptivas calculadas")

# GUARDAR RESULTADO EN CSV
info.to_csv(OUTPUT_FILE)                                 # GUARDAR CSV CON RESUMEN
if SHOW_INFO:
    print(f"[ GUARDADO ] Resumen inicial guardado en '{OUTPUT_FILE}'")
