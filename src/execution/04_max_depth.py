import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from math import log2
import json
import os

# --- PARÁMETROS ---
INPUT_CSV = '../../results/execution/00_contaminated.csv'
HIP_JSON = '../../results/execution/hiperparameters.json'

df = pd.read_csv(INPUT_CSV)

# Eliminar columna 'is_anomaly' si existe
if 'is_anomaly' in df.columns:
    df = df.drop(columns=['is_anomaly'])
    
Dat_np = df.select_dtypes(include=[np.number]).values.flatten()  # Convertir a 1D

# --- FUNCIONES ---
def contaminar_dat_5pct(Dat, porcentaje=0.05, incremento=0.5, random_state=None):
    np.random.seed(random_state)
    S = len(Dat)
    n_anom = max(1, int(S * porcentaje))
    Dat_contaminado = Dat.copy()
    indices_anom = np.random.choice(S, size=n_anom, replace=False)
    Dat_contaminado[indices_anom] = Dat_contaminado[indices_anom] * (1 + incremento)
    return Dat_contaminado, indices_anom

def ajustar_profundidad_maxima(Dat, S, β=0.2, random_state=None, max_iter=50):
    np.random.seed(random_state)
    Dat_cont, indices_anom = contaminar_dat_5pct(Dat[:S], porcentaje=0.05, incremento=0.5, random_state=random_state)
    print(f"[INFO] Dataset contaminado con {len(indices_anom)} anomalías")

    D_max = int(log2(S))
    D_min = 1
    D = D_max
    print(f"[INFO] D inicial: {D}")

    IF = IsolationForest(max_samples=S, max_features=1.0, n_estimators=100, random_state=random_state)
    IF.fit(Dat_cont.reshape(-1, 1))
    scores = -IF.decision_function(Dat_cont.reshape(-1, 1))
    n_anom = max(1, int(S * 0.05))
    top_indices = np.argsort(scores)[-n_anom:]
    CR = scores[top_indices]
    R_25, R_75 = np.percentile(CR, [25, 75])
    print(f"[INFO] R_25={R_25:.4f}, R_75={R_75:.4f}")

    iteration = 0
    while iteration < max_iter:
        iteration += 1
        print(f"[INFO] Iteración {iteration}: Probando factor D={D}")
        max_samples = max(1, int(S * D / D_max))
        IF = IsolationForest(max_samples=max_samples, max_features=1.0, n_estimators=100, random_state=random_state)
        IF.fit(Dat_cont.reshape(-1, 1))
        scores = -IF.decision_function(Dat_cont.reshape(-1, 1))
        top_indices = np.argsort(scores)[-n_anom:]
        CR = scores[top_indices]
        R = np.mean(CR)
        print(f"[INFO] Tasa promedio de aislamiento R={R:.4f}")

        new_D = D
        if R < R_75 and D > D_min:
            new_D = max(int(D * (1 - β)), D_min)
        elif R > R_25 and D < D_max:
            new_D = min(int(D * (1 + β)), D_max)

        if new_D == D:
            print(f"[INFO] Factor D ajustado encontrado: {D}")
            return D

        D = new_D

    print(f"[WARN] Máximo número de iteraciones alcanzado, factor D final: {D}")
    return D

# --- EJECUCIÓN ---
S = 256
D_ajustado = ajustar_profundidad_maxima(Dat_np, S, β=0.2, random_state=42)
print(f"[INFO] Factor D final ajustado: {D_ajustado}")

# --- JSON: crear o actualizar hiperparameters.json ---
if os.path.exists(HIP_JSON):
    with open(HIP_JSON, 'r', encoding='utf-8') as f:
        hip_data = json.load(f)
else:
    hip_data = {}

hip_data['D'] = {
    "value": D_ajustado,
    "description": "Factor controlling Isolation Forest effective depth",
    "adjustment_method": "Adjusted using average isolation rate R and quartiles",
    "default": "log2(S)"
}

with open(HIP_JSON, 'w', encoding='utf-8') as f:
    json.dump(hip_data, f, indent=4)

print(f"[INFO] hiperparameters.json actualizado con D={D_ajustado}")
