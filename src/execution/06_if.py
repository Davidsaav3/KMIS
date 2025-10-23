import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import os
import json

# --- RUTAS DE ARCHIVOS ---
RESULTS_FOLDER = '../../results'
EXECUTION_FOLDER = os.path.join(RESULTS_FOLDER, 'execution')
INPUT_CSV = os.path.join(EXECUTION_FOLDER, '00_contaminated.csv')
OUTPUT_CSV = os.path.join(EXECUTION_FOLDER, '06_global.csv')
OUTPUT_IF_CSV = os.path.join(EXECUTION_FOLDER, '06_global_if.csv')
HIP_JSON = os.path.join(EXECUTION_FOLDER, 'hiperparameters.json')

# --- OPCIONES ---
SAVE_ANOMALY_CSV = True
SORT_ANOMALY_SCORE = True
INCLUDE_SCORE = True
NORMALIZE_SCORE = True
SHOW_INFO = True
RANDOM_STATE = 42

# --- CARGAR HIPERPARÁMETROS ---
if os.path.exists(HIP_JSON):
    with open(HIP_JSON, 'r', encoding='utf-8') as f:
        hip_data = json.load(f)
else:
    hip_data = {}

# --- MAPEO DE HIPERPARÁMETROS ---
S = hip_data.get('S', {}).get('value', 'auto')         # Tamaño de muestra por árbol
T = hip_data.get('T', {}).get('value', 100)            # Número de árboles
F = hip_data.get('F', {}).get('value', 1.0)            # Máximo de características por árbol
D = hip_data.get('D', {}).get('value', None)           # Profundidad máxima
Th = hip_data.get('Th', {}).get('value', 0.01)         # Umbral de detección / contaminación

if SHOW_INFO:
    print(f"[INFO] Hiperparámetros cargados: S={S}, T={T}, F={F}, D={D}, Th={Th}")

# --- CARGAR DATASET GLOBAL ---
df = pd.read_csv(INPUT_CSV)
if SHOW_INFO:
    print(f"[INFO] Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")

# --- SEPARAR COLUMNA 'is_anomaly' ---
if 'is_anomaly' in df.columns:
    df_input = df.drop(columns=['is_anomaly'])
    is_anomaly_column = df['is_anomaly']
else:
    df_input = df.copy()
    is_anomaly_column = pd.Series([0]*len(df_input), name='is_anomaly')

# --- SELECCIONAR COLUMNAS NUMÉRICAS ---
num_cols = df_input.select_dtypes(include=['int64', 'float64']).columns.tolist()
df_num = df_input[num_cols]

# --- ESCALAR DATOS ---
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_num)

# --- ENTRENAR ISOLATION FOREST ---
clf_params = {
    "n_estimators": T,
    "max_samples": S,
    "contamination": Th,
    "max_features": F,
    "random_state": RANDOM_STATE,
    "n_jobs": -1
}
#if D is not None:
    #clf_params["max_depth"] = D

clf = IsolationForest(**clf_params)
clf.fit(df_scaled)

# --- CALCULAR SCORE Y PREDICCIÓN ---
anomaly_score = clf.decision_function(df_scaled) * -1  # Más positivo = más anómalo
pred = clf.predict(df_scaled)  # 1 = normal, -1 = anomalía

df['anomaly'] = np.where(pred == 1, 0, 1)
df['anomaly_score'] = anomaly_score
df['is_anomaly'] = is_anomaly_column

# --- INFORMACIÓN GENERAL ---
num_anomalies = df['anomaly'].sum()
num_normals = df.shape[0] - num_anomalies
if SHOW_INFO:
    print(f"[INFO] Total registros: {df.shape[0]}")
    print(f"[INFO] Anomalías detectadas: {num_anomalies}")
    print(f"[INFO] Registros normales: {num_normals}")
    print(f"[INFO] Porcentaje anomalías: {num_anomalies/df.shape[0]*100:.2f}%")

# --- GUARDAR CSV COMPLETO ---
df.to_csv(OUTPUT_CSV, index=False)
if SHOW_INFO:
    print(f"[GUARDADO] CSV completo con anomalías en '{OUTPUT_CSV}'")

# --- GUARDAR CSV SOLO CON ANOMALÍAS ---
if SAVE_ANOMALY_CSV:
    df_anomalies = df[df['anomaly'] == 1].copy()
    
    # NORMALIZAR SCORE ENTRE 0 Y 1
    if NORMALIZE_SCORE:
        min_score = df_anomalies['anomaly_score'].min()
        max_score = df_anomalies['anomaly_score'].max()
        if max_score > min_score:
            df_anomalies['anomaly_score'] = (df_anomalies['anomaly_score'] - min_score) / (max_score - min_score)
    
    # ORDENAR DE MAYOR A MENOR
    if SORT_ANOMALY_SCORE:
        df_anomalies = df_anomalies.sort_values(by='anomaly_score', ascending=False).reset_index(drop=True)
    
    # ELIMINAR SCORE SI NO SE INCLUYE
    if not INCLUDE_SCORE:
        df_anomalies.drop(columns=['anomaly_score'], inplace=True)
    
    df_anomalies.to_csv(OUTPUT_IF_CSV, index=False)
    if SHOW_INFO:
        print(f"[GUARDADO] CSV de anomalías {'ordenadas' if SORT_ANOMALY_SCORE else ''} en '{OUTPUT_IF_CSV}'")
