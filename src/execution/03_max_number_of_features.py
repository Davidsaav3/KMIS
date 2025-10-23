import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
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
def ajustar_numero_caracteristicas(Dat, F_inicial=1.0, α_reduccion=0.5, α_aumento=1.5, random_state=None):
    """Ajusta el número máximo de características F según varianza de submuestras"""
    np.random.seed(random_state)
    scaler = MinMaxScaler()
    Dat_norm = scaler.fit_transform(Dat)
    print("[INFO] Dataset normalizado a rango [0,1]")

    num_features = Dat.shape[1]
    F_min = 1 / num_features
    F_max = 1.0
    F = F_inicial
    print(f"[INFO] F inicial: {F:.4f}")

    while True:
        n_selected = max(1, int(F * num_features))
        selected_idx = np.random.choice(num_features, size=n_selected, replace=False)
        selected_features = Dat_norm[:, selected_idx]

        σ2 = np.var(selected_features, axis=0)
        V_prom = np.mean(σ2)
        Q1, Q2, Q3, Q4 = np.percentile(σ2, [25, 50, 75, 100])
        BQ1F = (np.max([Q1]) + np.min([Q2])) / 2
        BQ4F = (np.max([Q3]) + np.min([Q4])) / 2

        print(f"[INFO] V_prom={V_prom:.4f}, BQ1F={BQ1F:.4f}, BQ4F={BQ4F:.4f}, F={F:.4f}")

        if V_prom > BQ4F:
            if F * α_reduccion < F_min:
                print("[INFO] F alcanzó F_min, se retorna F")
                return F
            F *= α_reduccion
            print(f"[INFO] Reducción de F a {F:.4f}")
        elif V_prom < BQ1F:
            if F * α_aumento > F_max:
                print("[INFO] F alcanzó F_max, se retorna F")
                return F
            F *= α_aumento
            print(f"[INFO] Aumento de F a {F:.4f}")
        else:
            print("[INFO] F ajustado encontrado")
            return F

# --- EJECUCIÓN ---
F_ajustado = ajustar_numero_caracteristicas(Dat_np, F_inicial=1.0, α_reduccion=0.5, α_aumento=1.5, random_state=42)
print(f"[INFO] Número máximo de características final ajustado: {F_ajustado:.4f}")

# --- JSON: crear o actualizar hiperparameters.json ---
if os.path.exists(HIP_JSON):
    with open(HIP_JSON, 'r', encoding='utf-8') as f:
        hip_data = json.load(f)
else:
    hip_data = {}

hip_data['F'] = {
    "value": F_ajustado,
    "description": "Maximum number of features considered per tree",
    "adjustment_method": "Increment/decrement search based on variance",
    "default": 1.0
}

with open(HIP_JSON, 'w', encoding='utf-8') as f:
    json.dump(hip_data, f, indent=4)

print(f"[INFO] hiperparameters.json actualizado con F={F_ajustado:.4f}")
