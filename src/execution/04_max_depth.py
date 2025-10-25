import numpy as np 
import pandas as pd
from sklearn.ensemble import IsolationForest
from math import log2
import json
import os

# PARÁMETROS DE ENTRADA Y SALIDA
INPUT_CSV = '../../results/preparation/05_variance_recortado.csv'  # RUTA DEL CSV DE DATOS
HIP_JSON = '../../results/execution/hiperparameters.json'  # RUTA DEL JSON DE HIPERPARÁMETROS
json_path = "../../results/execution/hiperparameters.json"

# Cargar el JSON
with open(json_path, "r") as f:
    hiperparams = json.load(f)
S = hiperparams["S"]["value"]  

# CARGAR DATASET
df = pd.read_csv(INPUT_CSV)  # LEER CSV DESDE DISCO

# ELIMINAR COLUMNA 'is_anomaly' SI EXISTE
if 'is_anomaly' in df.columns:
    df = df.drop(columns=['is_anomaly'])  # ELIMINAR COLUMNA NO RELEVANTE

# CONVERTIR DATOS NUMÉRICOS A VECTOR 1D
Dat_np = df.select_dtypes(include=[np.number]).values.flatten()  # EXTRAER Y APLANAR DATOS NUMÉRICOS

# CONTAMINA 5% DE LOS DATOS AUMENTANDO SUS VALORES
def contaminar_dat_5pct(Dat, porcentaje=0.05, incremento=0.5, random_state=None):
    np.random.seed(random_state)
    S = len(Dat)
    n_anom = max(1, int(S * porcentaje))
    Dat_cont = Dat.copy()
    indices_anom = np.random.choice(S, size=n_anom, replace=False)
    Dat_cont[indices_anom] *= (1 + incremento)
    return Dat_cont, indices_anom


# [ MAIN MEJORADO ]
def ajustar_profundidad_maxima_mejorado(Dat, S, betha=0.03, random_state=None, max_iter=50, reps=5):
    # AJUSTA D SEGÚN LA TASA PROMEDIO DE AISLAMIENTO
    np.random.seed(random_state)  # FIJAR SEMILLA ALEATORIA

    # [ CAMBIO ] aumentar porcentaje de anomalías al 10%
    Dat_cont, indices_anom = contaminar_dat_5pct(Dat[:S], porcentaje=0.1, incremento=0.5, random_state=random_state)
    print(f"[INFO] Dataset contaminado con {len(indices_anom)} anomalías")

    # DEFINIR RANGO INICIAL DE PROFUNDIDAD
    D_max = int(log2(S))
    D_min = max(1, int(D_max * 0.25))
    D = max(D_min, int((D_min + D_max) / 2))  # [ CAMBIO ] iniciar D en la mediana
    print(f"[INFO] D inicial: {D}, D_min={D_min}, D_max={D_max}")

    # CALCULAR CUARTILES DE REFERENCIA
    IF = IsolationForest(max_samples=S, max_features=1.0, n_estimators=150, random_state=random_state)  # [ CAMBIO ] más árboles
    IF.fit(Dat_cont.reshape(-1, 1))
    scores = -IF.decision_function(Dat_cont.reshape(-1, 1))
    n_anom = max(1, int(S * 0.05))
    top_indices = np.argsort(scores)[-n_anom:]
    CR = scores[top_indices]
    R_25, R_75 = np.percentile(CR, [25, 75])
    print(f"[INFO] R_25={R_25:.4f}, R_75={R_75:.4f}")

    iteration = 0
    last_R = None
    stable_count = 0

    while iteration < max_iter:
        iteration += 1
        print(f"[INFO] Iteración {iteration}: Probando factor D={D}")
        max_samples = max(1, int(S * D / D_max))

        # [ CAMBIO ] promedio R sobre varias repeticiones
        R_list = []
        for r in range(reps):
            IF = IsolationForest(max_samples=max_samples, max_features=1.0, n_estimators=150,
                                 random_state=(random_state + r) if random_state is not None else None)
            IF.fit(Dat_cont.reshape(-1, 1))
            scores = -IF.decision_function(Dat_cont.reshape(-1, 1))
            top_indices = np.argsort(scores)[-n_anom:]
            R_list.append(np.mean(scores[top_indices]))
        R = np.mean(R_list)
        print(f"[INFO] Tasa promedio de aislamiento R={R:.4f}")

        # [ CAMBIO ] ajustar D con ceil para no caer a 1 demasiado rápido
        new_D = D
        if R < R_75 and D > D_min + 1:  # [ CAMBIO ] solo reducir si D > D_min+1
            new_D = max(int(np.ceil(D * (1 - betha))), D_min)  # REDUCIR D con ceil
        elif R >= R_25 * 0.95 and D < D_max:  # [ CAMBIO ] aumentar D más fácilmente
            new_D = min(int(np.ceil(D * (1 + betha))), D_max)  # AUMENTAR D con ceil

        # CRITERIO DE CONVERGENCIA
        if last_R is not None and abs(R - last_R) < 0.005:  # [ CAMBIO ] tolerancia para estabilidad
            stable_count += 1
            if stable_count >= 3:
                print(f"[INFO] Convergencia alcanzada. Factor D final: {D}")
                return D
        else:
            stable_count = 0

        if new_D == D:
            print(f"[INFO] Factor D ajustado encontrado: {D}")
            return D

        D = new_D
        last_R = R

    print(f"[WARN] Máximo número de iteraciones alcanzado, factor D final: {D}")
    return D


# LEER S DESDE JSON
if os.path.exists(HIP_JSON):
    with open(HIP_JSON, 'r', encoding='utf-8') as f:
        hip_data = json.load(f)
    S = hip_data.get('S', {}).get('value', 200)  # Si no existe, usar 200 como default
else:
    S = 256  # Default si no existe JSON


# CALL
# D_ajustado = ajustar_profundidad_maxima(Dat_np, S=S, betha=0.2, random_state=42, max_iter=50)
# D_ajustado = ajustar_profundidad_maxima(Dat_np, S, betha=0.2, random_state=42) # Ajustado        
D_ajustado = ajustar_profundidad_maxima_mejorado( Dat_np, S=S, betha=0.5, random_state=42, max_iter=10) # 
print(f"[FIN] hiperparameters.json actualizado con D={D_ajustado}")


# CARGAR O CREAR JSON DE HIPERPARÁMETROS
if os.path.exists(HIP_JSON):
    with open(HIP_JSON, 'r', encoding='utf-8') as f:
        hip_data = json.load(f)  # LEER JSON EXISTENTE
else:
    hip_data = {}  # CREAR NUEVO SI NO EXISTE
# ACTUALIZAR O AÑADIR VALOR DE D
hip_data['D'] = {
    "value": D_ajustado,  # VALOR FINAL DE D
    "description": "Factor controlling Isolation Forest effective depth",  # DESCRIPCIÓN
    "adjustment_method": "Adjusted using average isolation rate R and quartiles",  # MÉTODO DE AJUSTE
    "default": "log2(S) = 8 o 6"  # VALOR POR DEFECTO
}
# GUARDAR JSON ACTUALIZADO
with open(HIP_JSON, 'w', encoding='utf-8') as f:
    json.dump(hip_data, f, indent=4)  # ESCRIBIR JSON FORMATEADO