import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import os
import json
import math

# DEFINIR RUTAS DE ENTRADA Y SALIDA
RESULTS_FOLDER = '../../results'                          # CARPETA PRINCIPAL DE RESULTADOS
EXECUTION_FOLDER = os.path.join(RESULTS_FOLDER, 'execution')  # CARPETA DE EJECUCIÓN
INPUT_CSV = os.path.join(EXECUTION_FOLDER, '00_contaminated.csv')  # CSV DE ENTRADA
OUTPUT_CSV = os.path.join(EXECUTION_FOLDER, '06_global.csv')       # CSV COMPLETO CON ANOMALÍAS
OUTPUT_IF_CSV = os.path.join(EXECUTION_FOLDER, '06_global_if.csv') # CSV SOLO ANOMALÍAS
HIP_JSON = os.path.join(EXECUTION_FOLDER, 'hiperparameters.json')  # JSON DE HIPERPARÁMETROS

# CONFIGURACIÓN DE OPCIONES
SAVE_ANOMALY_CSV = True          # GUARDAR CSV SOLO CON ANOMALÍAS
SORT_ANOMALY_SCORE = True        # ORDENAR ANOMALÍAS POR SCORE
INCLUDE_SCORE = True             # INCLUIR SCORE EN CSV DE ANOMALÍAS
NORMALIZE_SCORE = True           # NORMALIZAR SCORE ENTRE 0 Y 1
SHOW_INFO = True                 # MOSTRAR INFO EN CONSOLA
RANDOM_STATE = 42                # SEMILLA ALEATORIA

# CARGAR HIPERPARÁMETROS DESDE JSON
if os.path.exists(HIP_JSON):
    with open(HIP_JSON, 'r', encoding='utf-8') as f:
        hip_data = json.load(f)
else:
    hip_data = {}

# ASIGNAR HIPERPARÁMETROS CON VALORES POR DEFECTO
S = hip_data.get('S', {}).get('value', 'auto')  # TAMAÑO DE MUESTRA POR ÁRBOL
T = hip_data.get('T', {}).get('value', 100)     # NÚMERO DE ÁRBOLES
F = hip_data.get('F', {}).get('value', 1.0)     # MÁXIMO DE CARACTERÍSTICAS POR ÁRBOL
D = hip_data.get('D', {}).get('value', None)    # PROFUNDIDAD MÁXIMA
Th = hip_data.get('Th', {}).get('value', 0.01)  # UMBRAL DE DETECCIÓN

if SHOW_INFO:
    print(f"[INFO] Hiperparámetros cargados: S={S}, T={T}, F={F}, D={D}, Th={Th}")

# CARGAR DATASET
df = pd.read_csv(INPUT_CSV)
if SHOW_INFO:
    print(f"[INFO] Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")

# SEPARAR COLUMNA DE ANOMALÍAS SI EXISTE
if 'is_anomaly' in df.columns:
    df_input = df.drop(columns=['is_anomaly'])       # ELIMINAR COLUMNA ORIGINAL PARA TRAINING
    is_anomaly_column = df['is_anomaly']            # GUARDAR COLUMNA ORIGINAL
else:
    df_input = df.copy()
    is_anomaly_column = pd.Series([0]*len(df_input), name='is_anomaly')  # COLUMNA DEFAULT

# SELECCIONAR SOLO COLUMNAS NUMÉRICAS
num_cols = df_input.select_dtypes(include=['int64', 'float64']).columns.tolist()
df_num = df_input[num_cols]

# ESCALAR DATOS
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_num)


# Entrenar un IF temporal solo para obtener los scores
# temp_if = IsolationForest(contamination='auto', random_state=42)
# temp_if.fit(df_scaled)
# Obtener puntuaciones de anomalía
# scores = temp_if.score_samples(df_scaled)  # más bajo = más anómalo
# Calcular fracción de registros bajo el umbral → contamination equivalente
# NOTA: ajustamos el signo porque score_samples devuelve valores negativos para anomalías
#contamination = np.mean(-scores >= Th)
#contamination = min(contamination, 0.5)  # max permitido por IF
#print(f"[INFO] Contamination equivalente a Th={Th}: {contamination:.4f}")

# INTEGRAR D EN MAX SAMPLES S
print(f"[INFO] S: {S}")
D_max = int(math.log2(S))
max_samples = max(1, int(S * D / D_max)) # reduce el número de muestras por árbol para que la profundidad efectiva sea aproximadamente D
print(f"[INFO] max_samples: {max_samples}")


# CONFIGURAR Y ENTRENAR ISOLATION FOREST
clf_params = {
    "n_estimators": T,
    "max_samples": max_samples,
    "contamination": 0.01,
    "max_features": F,
    "random_state": RANDOM_STATE, 
    "n_jobs": -1
}
# Scikit-learn calcula automáticamente la profundidad de los árboles hasta log2(max_samples) o hasta que se cumpla la condición de aislamiento de nodos.

clf = IsolationForest(**clf_params)
clf.fit(df_scaled)  # ENTRENAR MODELO

# CALCULAR SCORE Y PREDICCIÓN DE ANOMALÍAS
anomaly_score = clf.decision_function(df_scaled) * -1  # SCORE: MÁS ALTO = MÁS ANÓMALO
pred = clf.predict(df_scaled)                           # PREDICCIÓN: 1=NORMAL, -1=ANOMALÍA
df['anomaly'] = np.where(pred == 1, 0, 1)              # CREAR COLUMNA ANOMALÍAS
df['anomaly_score'] = anomaly_score                    # AÑADIR SCORE
df['is_anomaly'] = is_anomaly_column                   # RESTAURAR COLUMNA ORIGINAL

# MOSTRAR INFORMACIÓN GENERAL
num_anomalies = df['anomaly'].sum()
num_normals = df.shape[0] - num_anomalies
if SHOW_INFO:
    print(f"[INFO] Total registros: {df.shape[0]}")
    print(f"[INFO] Anomalías detectadas: {num_anomalies}")
    print(f"[INFO] Registros normales: {num_normals}")
    print(f"[INFO] Porcentaje anomalías: {num_anomalies/df.shape[0]*100:.2f}%")

# GUARDAR CSV COMPLETO CON TODAS LAS COLUMNAS
df.to_csv(OUTPUT_CSV, index=False)
if SHOW_INFO:
    print(f"[GUARDADO] CSV completo con anomalías en '{OUTPUT_CSV}'")

# GUARDAR CSV SOLO CON ANOMALÍAS
if SAVE_ANOMALY_CSV:
    df_anomalies = df[df['anomaly'] == 1].copy()
    
    # NORMALIZAR SCORE ENTRE 0 Y 1
    if NORMALIZE_SCORE:
        min_score = df_anomalies['anomaly_score'].min()
        max_score = df_anomalies['anomaly_score'].max()
        if max_score > min_score:
            df_anomalies['anomaly_score'] = (df_anomalies['anomaly_score'] - min_score) / (max_score - min_score)
    
    # ORDENAR ANOMALÍAS POR SCORE DE MAYOR A MENOR
    if SORT_ANOMALY_SCORE:
        df_anomalies = df_anomalies.sort_values(by='anomaly_score', ascending=False).reset_index(drop=True)
    
    # ELIMINAR SCORE SI NO SE DESEA INCLUIR
    if not INCLUDE_SCORE:
        df_anomalies.drop(columns=['anomaly_score'], inplace=True)
    
    # GUARDAR CSV DE ANOMALÍAS
    df_anomalies.to_csv(OUTPUT_IF_CSV, index=False)
    if SHOW_INFO:
        print(f"[GUARDADO] CSV de anomalías {'ordenadas' if SORT_ANOMALY_SCORE else ''} en '{OUTPUT_IF_CSV}'")
