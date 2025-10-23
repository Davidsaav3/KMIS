import numpy as np
import pandas as pd
import json
import os

# --- PARÁMETROS ---
INPUT_CSV = '../../results/execution/00_contaminated.csv'
HIP_JSON = '../../results/execution/hiperparameters.json'

df = pd.read_csv(INPUT_CSV)

# Eliminar columna 'is_anomaly' si existe
if 'is_anomaly' in df.columns:
    df = df.drop(columns=['is_anomaly'])
    
Dat_np = df.select_dtypes(include=[np.number]).values.flatten()  # 1D

# --- FUNCIONES ---
def contaminar_dat_5pct(Dat, porcentaje=0.05, incremento=0.5, random_state=None):
    np.random.seed(random_state)
    S = len(Dat)
    n_anom = max(1, int(S * porcentaje))
    Dat_contaminado = Dat.copy()
    indices_anom = np.random.choice(S, size=n_anom, replace=False)
    Dat_contaminado[indices_anom] = Dat_contaminado[indices_anom] * (1 + incremento)
    return Dat_contaminado, indices_anom

def calcular_FC(Dat_cont, indices_anom, Th, delta=0.2):
    pred = np.zeros(len(Dat_cont))
    pred[Dat_cont >= Th] = 1
    y_true = np.zeros(len(Dat_cont))
    y_true[indices_anom] = 1
    FP = np.sum((pred == 1) & (y_true == 0)) / max(1, np.sum(y_true == 0))
    FN = np.sum((pred == 0) & (y_true == 1)) / max(1, np.sum(y_true == 1))
    FC = delta * FP + (1 - delta) * FN
    return FC

def ajustar_umbral(Dat, delta=0.2, Th_min=0.0, Th_max=1.0, grad=0.01, Th=0.5, random_state=None):
    np.random.seed(random_state)
    Dat_cont, indices_anom = contaminar_dat_5pct(Dat, porcentaje=0.05, incremento=0.5, random_state=random_state)
    print(f"[INFO] Dataset contaminado con {len(indices_anom)} anomalías")

    while Th_max - Th_min >= grad:
        mid1 = (Th + Th_min) / 2
        mid2 = (Th_max + Th) / 2

        FC1 = calcular_FC(Dat_cont, indices_anom, mid1, delta)
        FC2 = calcular_FC(Dat_cont, indices_anom, mid2, delta)

        print(f"[INFO] Th={Th:.4f}, mid1={mid1:.4f}, FC1={FC1:.4f}, mid2={mid2:.4f}, FC2={FC2:.4f}")

        if FC1 < FC2:
            Th_max = Th
            Th = mid1
        else:
            Th_min = Th
            Th = mid2

    print(f"[INFO] Umbral de detección ajustado: Th={Th:.4f}")
    return Th

# --- EJECUCIÓN ---
Th_ajustado = ajustar_umbral(Dat_np, delta=0.2, Th_min=0.0, Th_max=1.0, grad=0.01, Th=0.5, random_state=42)

# --- JSON: crear o actualizar hiperparameters.json ---
if os.path.exists(HIP_JSON):
    with open(HIP_JSON, 'r', encoding='utf-8') as f:
        hip_data = json.load(f)
else:
    hip_data = {}

hip_data['Th'] = {
    "value": Th_ajustado,
    "description": "Detection threshold to classify anomalies",
    "adjustment_method": "Binary search minimizing cost function (FP,FN)",
    "default": 1.0
}

with open(HIP_JSON, 'w', encoding='utf-8') as f:
    json.dump(hip_data, f, indent=4)

print(f"[INFO] hiperparameters.json actualizado con Th={Th_ajustado}")
