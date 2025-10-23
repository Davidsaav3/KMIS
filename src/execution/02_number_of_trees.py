import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score
import json
import os

# --- PARÁMETROS ---
INPUT_CSV = '../../results/execution/00_contaminated.csv'
HIP_JSON = '../../results/execution/hiperparameters.json'

df = pd.read_csv(INPUT_CSV)

# Eliminar columna 'is_anomaly' si existe
if 'is_anomaly' in df.columns:
    df = df.drop(columns=['is_anomaly'])
    
Dat_np = df.select_dtypes(include=[np.number]).values  # Convertir a numpy array 2D

# --- FUNCIONES ---
def contaminar_dat(Dat, S, porcentaje=0.01, incremento=0.5, random_state=None):
    """Contamina un subconjunto del dataset con anomalías artificiales"""
    np.random.seed(random_state)
    indices_muestra = np.random.choice(Dat.shape[0], size=S, replace=False)
    muestra = Dat[indices_muestra, :].copy()
    n_anom = max(1, int(S * porcentaje))
    indices_anom = np.random.choice(S, size=n_anom, replace=False)
    col = np.random.randint(muestra.shape[1])
    muestra[indices_anom, col] *= (1 + incremento)
    return muestra, indices_anom, indices_muestra

def ajustar_numero_arboles(Dat, S, T_min=5, T_max=100, step=5, N=3, F1sta=0.01, random_state=None):
    """Ajusta el número de árboles (T) según estabilidad del F1-score"""
    np.random.seed(random_state)
    Dat_cont, indices_anom_real, _ = contaminar_dat(Dat, S, porcentaje=0.01,
                                                    incremento=0.5, random_state=random_state)
    print(f"[INFO] Dataset contaminado con {len(indices_anom_real)} anomalías artificiales")
    T = T_min
    F1_list = []

    while T <= T_max:
        print(f"[INFO] Probando T={T} árboles")
        IF = IsolationForest(n_estimators=T, contamination=0.01, random_state=random_state)
        IF.fit(Dat_cont)
        scores = -IF.decision_function(Dat_cont)
        n_anom_pred = max(1, int(S * 0.01))
        indices_pred = np.argsort(scores)[-n_anom_pred:]

        y_true = np.zeros(S)
        y_true[indices_anom_real] = 1
        y_pred = np.zeros(S)
        y_pred[indices_pred] = 1
        F1 = f1_score(y_true, y_pred)
        F1_list.append(F1)
        print(f"[INFO] F1-score: {F1:.4f}")

        if len(F1_list) >= N:
            cumple = sum(1 for f in F1_list[-N:] if f <= F1sta)
            if cumple == N:
                print(f"[INFO] F1-score estable detectado. T final: {T}")
                return T

        T += step

    print(f"[INFO] F1-score no estabilizó. T final = {T_max}")
    return T_max

# --- AJUSTE ---
S = 200  # Se puede usar el valor de S ajustado previamente
T_ajustado = ajustar_numero_arboles(Dat_np, S, T_min=5, T_max=100, step=5, N=3, F1sta=0.01, random_state=42)
print(f"[INFO] Número de árboles final ajustado: {T_ajustado}")

# --- JSON: crear o actualizar hiperparameters.json ---
if os.path.exists(HIP_JSON):
    with open(HIP_JSON, 'r', encoding='utf-8') as f:
        hip_data = json.load(f)
else:
    hip_data = {}

hip_data['T'] = {
    "value": T_ajustado,
    "description": "Number of trees in the Isolation Forest",
    "adjustment_method": "F1-score stabilization over iterations",
    "default": 100
}

with open(HIP_JSON, 'w', encoding='utf-8') as f:
    json.dump(hip_data, f, indent=4)

print(f"[INFO] hiperparameters.json actualizado con T={T_ajustado}")
