import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import os
import numpy as np
import json

# --- PARÁMETROS GENERALES ---
RESULTS_FOLDER = '../../results'
EXECUTION_FOLDER = os.path.join(RESULTS_FOLDER, 'execution')
INPUT_CSV = os.path.join(EXECUTION_FOLDER, '00_contaminated.csv')
OUTPUT_CSV = os.path.join(EXECUTION_FOLDER, '06_global.csv')
OUTPUT_IF_CSV = os.path.join(EXECUTION_FOLDER, '06_global_if.csv')

SAVE_ANOMALY_CSV = True
SORT_ANOMALY_SCORE = True
INCLUDE_SCORE = True
NORMALIZE_SCORE = True
SHOW_INFO = True

HIP_JSON = os.path.join(EXECUTION_FOLDER, 'hiperparameters.json')

# --- CARGAR HIPERPARÁMETROS ---
if os.path.exists(HIP_JSON):
    with open(HIP_JSON, 'r', encoding='utf-8') as f:
        hip_data = json.load(f)
else:
    hip_data = {}

N_ESTIMATORS = hip_data.get('N_ESTIMATORS', {}).get('value', 100)
MAX_SAMPLES = hip_data.get('S', {}).get('value', 'auto')
CONTAMINATION = hip_data.get('CONTAMINATION', {}).get('value', 0.01)
MAX_FEATURES = hip_data.get('MAX_FEATURES', {}).get('value', 1.0)
BOOTSTRAP = hip_data.get('BOOTSTRAP', {}).get('value', False)
N_JOBS = hip_data.get('N_JOBS', {}).get('value', -1)
RANDOM_STATE = hip_data.get('RANDOM_STATE', {}).get('value', 42)
VERBOSE = hip_data.get('VERBOSE', {}).get('value', 0)

if SHOW_INFO:
    print(f"[INFO] Hiperparámetros cargados desde '{HIP_JSON}'")

# --- CARGAR DATASET GLOBAL ---
df = pd.read_csv(INPUT_CSV)
if SHOW_INFO:
    print(f"[INFO] Dataset global cargado: {df.shape[0]} filas, {df.shape[1]} columnas")

# SEPARAR COLUMNA 'is_anomaly'
if 'is_anomaly' in df.columns:
    df_input = df.drop(columns=['is_anomaly'])
    is_anomaly_column = df['is_anomaly']
else:
    df_input = df.copy()
    is_anomaly_column = pd.Series([0]*len(df_input), name='is_anomaly')

# SELECCIONAR COLUMNAS NUMÉRICAS
num_cols = df_input.select_dtypes(include=['int64', 'float64']).columns.tolist()

# ESCALAR DATOS
scaler = StandardScaler()
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
anomaly_score = clf.decision_function(df_scaled) * -1
pred = clf.predict(df_scaled)
df['anomaly'] = np.where(pred == 1, 0, 1)
df['anomaly_score'] = anomaly_score
df['is_anomaly'] = is_anomaly_column

# INFORMACIÓN GENERAL
num_anomalies = df['anomaly'].sum()
num_normals = df.shape[0] - num_anomalies
if SHOW_INFO:
    print(f"[INFO] Registros totales (GLOBAL): {df.shape[0]}")
    print(f"[INFO] Anomalías detectadas (GLOBAL): {num_anomalies}")
    print(f"[INFO] Registros normales (GLOBAL): {num_normals}")
    print(f"[INFO] Porcentaje anomalías (GLOBAL): {num_anomalies/df.shape[0]*100:.2f}%")

# GUARDAR CSV COMPLETO
df.to_csv(OUTPUT_CSV, index=False)
if SHOW_INFO:
    print(f"[GUARDADO] CSV completo con anomalías en '{OUTPUT_CSV}'")

# GUARDAR CSV SOLO CON ANOMALÍAS
if SAVE_ANOMALY_CSV:
    df_anomalies = df.loc[df['anomaly'] == 1].copy()
    if NORMALIZE_SCORE:
        min_score = df_anomalies['anomaly_score'].min()
        max_score = df_anomalies['anomaly_score'].max()
        if max_score > min_score:
            df_anomalies['anomaly_score'] = (df_anomalies['anomaly_score'] - min_score) / (max_score - min_score)
    if SORT_ANOMALY_SCORE:
        df_anomalies = df_anomalies.sort_values(by='anomaly_score', ascending=False).reset_index(drop=True)
    if not INCLUDE_SCORE:
        df_anomalies.drop(columns=['anomaly_score'], inplace=True)
    df_anomalies.to_csv(OUTPUT_IF_CSV, index=False)
    if SHOW_INFO:
        print(f"[GUARDADO] CSV anomalías {'ordenadas' if SORT_ANOMALY_SCORE else ''} en '{OUTPUT_IF_CSV}'")
