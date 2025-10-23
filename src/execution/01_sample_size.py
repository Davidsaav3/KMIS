import pandas as pd
import numpy as np
import json
import os

# --- Parámetros ---
INPUT_CSV = '../../results/execution/00_contaminated.csv'
HIP_JSON = '../../results/execution/hiperparameters.json'

# --- Cargar dataset ---
df = pd.read_csv(INPUT_CSV)

# Eliminar columna 'is_anomaly' si existe
if 'is_anomaly' in df.columns:
    df = df.drop(columns=['is_anomaly'])

data = df.select_dtypes(include=[np.number]).values.flatten()  # Aplanar a 1D

# --- Funciones ---
def desviacion_tipica(data):
    """Calcula la desviación típica de un vector 1D"""
    return np.std(data)

def ajustar_tamano_muestra(Dat, S_inicial=256, e_sigma=0.05, IncDat=0.1, random_state=None):
    np.random.seed(random_state)
    S = S_inicial
    print(f"[INFO] Tamaño inicial de muestra: {S}")

    sigma_o = desviacion_tipica(Dat)
    print(f"[INFO] Desviación típica del conjunto original sigma_o: {sigma_o:.4f}")

    sigma_min = desviacion_tipica(np.random.choice(Dat, size=min(S, len(Dat)), replace=False))
    print(f"[INFO] Desviación típica de la muestra inicial sigma_min: {sigma_min:.4f}")

    while S < len(Dat) and not (sigma_o - e_sigma <= sigma_min <= sigma_o + e_sigma):
        S = int(min(S * (1 + IncDat), len(Dat)))
        print(f"[INFO] Incrementando tamaño de muestra a: {S}")

        muestra = np.random.choice(Dat, size=S, replace=False)
        sigma_min = desviacion_tipica(muestra)
        print(f"[INFO] Desviación típica de la nueva muestra sigma_min: {sigma_min:.4f}")

    print(f"[INFO] Tamaño de muestra final ajustado: {S}")
    return S

# --- Ajuste ---
S_ajustado = ajustar_tamano_muestra(data, S_inicial=256, e_sigma=0.05, IncDat=0.1, random_state=42)

# --- Leer o crear hiperparameters.json ---
if os.path.exists(HIP_JSON):
    with open(HIP_JSON, 'r', encoding='utf-8') as f:
        hip_data = json.load(f)
else:
    hip_data = {}

# --- Actualizar información de S ---
hip_data['S'] = {
    "value": S_ajustado,
    "description": "Sample size for Isolation Forest",
    "adjustment_method": "Standard deviation based incremental sampling",
    "default": 256
}

# --- Guardar JSON ---
with open(HIP_JSON, 'w', encoding='utf-8') as f:
    json.dump(hip_data, f, indent=4)

print(f"[INFO] hiperparameters.json actualizado con S={S_ajustado}")
