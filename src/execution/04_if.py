import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import glob
import os
import numpy as np

# PARÁMETROS GENERALES
RESULTS_FOLDER = '../../results'  # Carpeta principal de resultados
EXECUTION_FOLDER = os.path.join(RESULTS_FOLDER, 'execution')  # Carpeta de ejecución
CLUSTER_PATTERN = 'cluster_*.csv'  # Patrón de archivos de clusters
INPUT_CSV = '../../results/execution/03_global.csv'     # ARCHIVO GLOBAL DE ENTRADA
OUTPUT_CSV = os.path.join(EXECUTION_FOLDER, '04_global.csv')  # SALIDA IF GLOBAL
OUTPUT_IF_CSV = os.path.join(EXECUTION_FOLDER, '04_global_if.csv')   # SALIDA IF GLOBAL SOLO ANOMALÍAS

SAVE_ANOMALY_CSV = True  # Guardar solo anomalías detectadas
SORT_ANOMALY_SCORE = True  # Ordenar CSV de anomalías por score (más anómalas arriba)
INCLUDE_SCORE = True  # Incluir columna 'anomaly_score' en CSV de anomalías
NORMALIZE_SCORE = True  # Activar/desactivar normalización de anomaly_score entre 0 y 1

# HIPERPARÁMETROS ISOLATION FOREST
N_ESTIMATORS = 100
# NÚMERO DE ÁRBOLES EN EL BOSQUE
# MÁS ÁRBOLES = MODELO MÁS ESTABLE Y PRECISO, PERO MÁS LENTO
MAX_SAMPLES = 'auto'
# NÚMERO DE MUESTRAS POR ÁRBOL
# 'auto' USA TODAS LAS FILAS
# MENOS MUESTRAS = ENTRENAMIENTO MÁS RÁPIDO, PERO MENOS ROBUSTO
CONTAMINATION = 0.01
# FRACCIÓN ESTIMADA DE ANOMALÍAS
# AJUSTA UMBRAL PARA CLASIFICAR ANOMALÍAS
# RECOMENDADO: 0.01-0.05 SEGÚN EXPECTATIVA DE ANOMALÍAS
MAX_FEATURES = 1.0
# PROPORCIÓN DE CARACTERÍSTICAS A USAR POR ÁRBOL
# MENOS CARACTERÍSTICAS = MÁS VARIABILIDAD ENTRE ÁRBOLES

BOOTSTRAP = False
# TRUE = MUESTREO CON REPETICIÓN POR ÁRBOL
# FALSE = MUESTRA SIN REPETICIÓN, MÁS PRECISO
N_JOBS = -1
# NÚMERO DE HILOS PARA ENTRENAMIENTO
# -1 USA TODOS LOS HILOS DISPONIBLES
RANDOM_STATE = 42
# SEMILLA PARA REPRODUCIBILIDAD DE RESULTADOS
VERBOSE = 0
# NIVEL DE VERBOSIDAD DEL MODELO
# 0 = SILENCIOSO, >0 = INFORMACIÓN DETALLADA DURANTE ENTRENAMIENTO
SHOW_INFO = True
# TRUE = MOSTRAR INFORMACIÓN DEL PROCESO EN CONSOLA


# ISOLATION FOREST GLOBAL
# CARGAR DATASET GLOBAL
df = pd.read_csv(INPUT_CSV)
if SHOW_INFO:
    print(f"[ INFO ] DATASET GLOBAL CARGADO: {df.shape[0]} FILAS, {df.shape[1]} COLUMNAS")

# SEPARAR COLUMNA 'is_anomaly'
if 'is_anomaly' in df.columns:
    df_input = df.drop(columns=['is_anomaly'])  # NO USAR EN ENTRENAMIENTO
    is_anomaly_column = df['is_anomaly']       # GUARDAR PARA COMPARACIÓN
else:
    df_input = df.copy()
    is_anomaly_column = pd.Series([0]*len(df_input), name='is_anomaly')  # COLUMNA TEMPORAL

# SELECCIONAR COLUMNAS NUMÉRICAS
num_cols = df_input.select_dtypes(include=['int64', 'float64']).columns.tolist()

# ESCALAR DATOS
scaler = StandardScaler()  # EVITA DOMINANCIA DE MAGNITUDES
df_scaled = scaler.fit_transform(df_input[num_cols])

# ENTRENAR ISOLATION FOREST GLOBAL
clf = IsolationForest(
    n_estimators=N_ESTIMATORS,
    max_samples=MAX_SAMPLES,
    contamination=CONTAMINATION,
    max_features=MAX_FEATURES,
    bootstrap=BOOTSTRAP,
    n_jobs=N_JOBS,
    random_state=RANDOM_STATE,
    verbose=VERBOSE
)
clf.fit(df_scaled)

# CALCULAR SCORE Y PREDICCIÓN
anomaly_score = clf.decision_function(df_scaled) * -1  # MÁS POSITIVO = MÁS ANÓMALO
pred = clf.predict(df_scaled)  # 1 = NORMAL, -1 = ANOMALÍA
df['anomaly'] = np.where(pred == 1, 0, 1)  # 0=normal, 1=anomalía
df['anomaly_score'] = anomaly_score
df['is_anomaly'] = is_anomaly_column

# INFORMACIÓN GENERAL
num_anomalies = df['anomaly'].sum()
num_normals = df.shape[0] - num_anomalies
if SHOW_INFO:
    print(f"[ INFO ] REGISTROS TOTALES (GLOBAL): {df.shape[0]}")
    print(f"[ INFO ] ANOMALÍAS DETECTADAS (GLOBAL): {num_anomalies}")
    print(f"[ INFO ] REGISTROS NORMALES (GLOBAL): {num_normals}")
    print(f"[ INFO ] PORCENTAJE ANOMALÍAS (GLOBAL): {num_anomalies/df.shape[0]*100:.2f}%")

# GUARDAR CSV COMPLETO
df.to_csv(OUTPUT_CSV, index=False)
if SHOW_INFO:
    print(f"[ GUARDADO ] CSV COMPLETO CON ANOMALÍAS EN '{OUTPUT_CSV}'")

# GUARDAR CSV SOLO CON ANOMALÍAS
if SAVE_ANOMALY_CSV:
    df_anomalies = df.loc[df['anomaly'] == 1].copy()
    df_anomalies['anomaly_score'] = df_anomalies['anomaly_score'].astype(float)

    # NORMALIZAR SCORE ENTRE 0 Y 1 SI SE ACTIVA
    if NORMALIZE_SCORE:
        min_score = df_anomalies['anomaly_score'].min()
        max_score = df_anomalies['anomaly_score'].max()
        if max_score > min_score:  # EVITAR DIVISIÓN POR CERO
            df_anomalies['anomaly_score'] = (df_anomalies['anomaly_score'] - min_score) / (max_score - min_score)

    # ORDENAR DE MAYOR A MENOR
    if SORT_ANOMALY_SCORE:
        df_anomalies = df_anomalies.sort_values(by='anomaly_score', ascending=False).reset_index(drop=True)

    # ELIMINAR SCORE SI NO SE INCLUYE
    if not INCLUDE_SCORE:
        df_anomalies.drop(columns=['anomaly_score'], inplace=True)

    # GUARDAR CSV FINAL
    df_anomalies.to_csv(OUTPUT_IF_CSV, index=False)
    if SHOW_INFO:
        print(f"[ GUARDADO ] CSV ANOMALÍAS {'ORDENADAS' if SORT_ANOMALY_SCORE else ''} EN '{OUTPUT_IF_CSV}'")


# ISOLATION FOREST POR CLUSTER
# LISTAR ARCHIVOS DE CLUSTERS
files = glob.glob(os.path.join(EXECUTION_FOLDER, CLUSTER_PATTERN))
if SHOW_INFO:
    print(f"[ INFO ] ARCHIVOS ENCONTRADOS PARA IF POR CLUSTER: {len(files)}")

# PROCESAR CADA ARCHIVO
for file_path in files:
    # Cargar CSV
    df = pd.read_csv(file_path)
    if SHOW_INFO:
        print(f"[ INFO ] PROCESANDO {file_path}: {df.shape[0]} FILAS, {df.shape[1]} COLUMNAS")

    # SEPARAR COLUMNA 'is_anomaly'
    if 'is_anomaly' in df.columns:
        df_input = df.drop(columns=['is_anomaly'])  # No usar en entrenamiento
        is_anomaly_column = df['is_anomaly']  # Guardar para comparación posterior
    else:
        df_input = df.copy()
        is_anomaly_column = pd.Series([0]*len(df_input), name='is_anomaly')  # Crear columna temporal

    # SELECCIONAR COLUMNAS NUMÉRICAS
    num_cols = df_input.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if len(num_cols) == 0:
        if SHOW_INFO:
            print("[ SKIP ] NO HAY COLUMNAS NUMÉRICAS PARA APLICAR IF")
        continue
    if SHOW_INFO:
        print(f"[ INFO ] NÚMERO DE COLUMNAS NUMÉRICAS: {len(num_cols)}")

    # ESCALAR DATOS
    scaler = StandardScaler()  # Evitar que columnas con gran magnitud dominen la detección
    df_scaled = scaler.fit_transform(df_input[num_cols])  # Transformación estándar (media=0, std=1)

    # CONFIGURAR ISOLATION FOREST
    clf = IsolationForest(
        n_estimators=N_ESTIMATORS,
        max_samples=MAX_SAMPLES,
        contamination=CONTAMINATION,
        max_features=MAX_FEATURES,
        bootstrap=BOOTSTRAP,
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE,
        verbose=VERBOSE
    )

    # ENTRENAMIENTO Y PREDICCIÓN
    clf.fit(df_scaled)  # Entrenar bosque de aislamiento
    anomaly_score = clf.decision_function(df_scaled) * -1  # Score: más positivo = más anómalo
    pred = clf.predict(df_scaled)  # Predicción de anomalías
    df['anomaly'] = np.where(pred == 1, 0, 1)  # Convertir a 0=normal, 1=anomalía

    # Añadir score y columna original de referencia
    df['anomaly_score'] = anomaly_score
    df['is_anomaly'] = is_anomaly_column

    # INFORMACIÓN GENERAL SOBRE ANOMALÍAS
    num_anomalies = df['anomaly'].sum()
    num_normals = df.shape[0] - num_anomalies
    if SHOW_INFO:
        print(f"[ INFO ] REGISTROS TOTALES: {df.shape[0]}")
        print(f"[ INFO ] ANOMALÍAS DETECTADAS: {num_anomalies}")
        print(f"[ INFO ] REGISTROS NORMALES: {num_normals}")
        print(f"[ INFO ] PORCENTAJE ANOMALÍAS: {num_anomalies/df.shape[0]*100:.2f}%")

    # GUARDAR CSV COMPLETO
    df.to_csv(file_path, index=False)
    if SHOW_INFO:
        print(f"[ GUARDADO ] RESULTADOS IF COMPLETOS EN {file_path}")

    # GUARDAR CSV SOLO CON ANOMALÍAS
    if SAVE_ANOMALY_CSV:
        df_anomalies = df[df['anomaly'] == 1].copy()  # Filtrar solo anomalías
        df_anomalies['anomaly_score'] = df_anomalies['anomaly_score'].astype(float)

        # Normalizar anomaly_score entre 0 y 1 si NORMALIZE_SCORE está activado
        if NORMALIZE_SCORE:
            min_score = df_anomalies['anomaly_score'].min()
            max_score = df_anomalies['anomaly_score'].max()
            if max_score > min_score:  # Evitar división por cero
                df_anomalies['anomaly_score'] = (df_anomalies['anomaly_score'] - min_score) / (max_score - min_score)

        # Ordenar de mayor a menor si está activado
        if SORT_ANOMALY_SCORE:
            df_anomalies = df_anomalies.sort_values(
                by='anomaly_score',
                ascending=False
            ).reset_index(drop=True)

        # Eliminar score si no se desea incluir
        if not INCLUDE_SCORE:
            df_anomalies.drop(columns=['anomaly_score'], inplace=True)

        # Guardar CSV de anomalías
        output_anomaly_csv = file_path.replace('.csv', '_if.csv')
        df_anomalies.to_csv(output_anomaly_csv, index=False)
        if SHOW_INFO:
            print(f"[ GUARDADO ] CSV DE ANOMALÍAS ORDENADAS EN {output_anomaly_csv}")

# BORRADO 03_GLOBAL.CSV
if os.path.exists(INPUT_CSV):
    os.remove(INPUT_CSV)
    print(f"[ OK ] Archivo eliminado: {INPUT_CSV}")
else:
    print(f"[ INFO ] No se encontró el archivo: {INPUT_CSV}")