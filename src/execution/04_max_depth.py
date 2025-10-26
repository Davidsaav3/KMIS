import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from math import log2
import json
import os

# RUTAS DE ENTRADA Y SALIDA
INPUT_CSV = '../../results/preparation/05_variance_recortado.csv'  # CSV DE DATOS
HIP_JSON = '../../results/execution/hiperparameters.json'  # JSON HIPERPARÁMETROS
json_path = "../../results/execution/hiperparameters.json"

# CARGAR JSON DE HIPERPARÁMETROS
with open(json_path, "r") as f:
    hiperparams = json.load(f)
S = hiperparams["S"]["value"]  # LEER VALOR DE S

# CARGAR DATASET
df = pd.read_csv(INPUT_CSV)  # LEER CSV

# ELIMINAR COLUMNA 'is_anomaly' SI EXISTE
if 'is_anomaly' in df.columns:
    df = df.drop(columns=['is_anomaly'])  # BORRAR COLUMNA NO RELEVANTE

# EXTRAER DATOS NUMÉRICOS A VECTOR 1D
Dat_np = df.select_dtypes(include=[np.number]).values.flatten()  # DATOS NUMÉRICOS APLANADOS


# [ CONTAMINAR 5% ] 
def contaminar_dat_5pct(Dat, porcentaje=0.05, incremento=0.5, random_state=None):
    np.random.seed(random_state)  # FIJAR SEMILLA
    S = len(Dat)
    n_anom = max(1, int(S * porcentaje))  # NUMERO DE ANOMALIAS
    Dat_cont = Dat.copy()  # COPIAR DATOS
    indices_anom = np.random.choice(S, size=n_anom, replace=False)  # INDICES ANOMALOS
    Dat_cont[indices_anom] *= (1 + incremento)  # AUMENTAR VALORES
    return Dat_cont, indices_anom


# [ MAIN ]
def ajustar_profundidad_maxima(Dat, S, betha=0.2, random_state=None, max_iter=50):
    # AJUSTA D SEGÚN LA TASA PROMEDIO DE AISLAMIENTO
    np.random.seed(random_state)  # FIJAR SEMILLA ALEATORIA
    # CONTAMINAR SUBMUESTRA CON ANOMALÍAS
    Dat_cont, indices_anom = contaminar_dat_5pct(Dat[:S], porcentaje=0.05, incremento=0.5, random_state=random_state)
    print(f"[INFO] Dataset contaminado con {len(indices_anom)} anomalías")

    # DEFINIR RANGO INICIAL DE PROFUNDIDAD
    D_max = int(log2(S))  # PROFUNDIDAD MÁXIMA TEÓRICA
    D_min = 1  # PROFUNDIDAD MÍNIMA PERMITIDA
    D = D_max  # INICIALIZAR CON D_MAX
    print(f"[INFO] D inicial: {D}")

    # CALCULAR REFERENCIA DE AISLAMIENTO CON D INICIAL
    IF = IsolationForest(max_samples=S, max_features=1.0, n_estimators=100, random_state=random_state)
    IF.fit(Dat_cont.reshape(-1, 1))  # ENTRENAR MODELO CON DATOS CONTAMINADOS
    scores = -IF.decision_function(Dat_cont.reshape(-1, 1))  # MAYOR SCORE = MÁS ANÓMALO
    n_anom = max(1, int(S * 0.05))  # CANTIDAD DE ANOMALÍAS A CONSIDERAR
    top_indices = np.argsort(scores)[-n_anom:]  # ÍNDICES DE MAYOR SCORE
    CR = scores[top_indices]  # SCORES DE ANOMALÍAS DETECTADAS
    R_25, R_75 = np.percentile(CR, [25, 75])  # CALCULAR CUARTILES
    print(f"[INFO] R_25={R_25:.4f}, R_75={R_75:.4f}")

    # INICIAR BUCLE DE AJUSTE ITERATIVO
    iteration = 0
    while iteration < max_iter:
        iteration += 1
        print(f"[INFO] Iteración {iteration}: Probando factor D={D}")
        max_samples = max(1, int(S * D / D_max))  # AJUSTAR TAMAÑO DE MUESTRA SEGÚN D
        IF = IsolationForest(max_samples=max_samples, max_features=1.0, n_estimators=100, random_state=random_state)
        IF.fit(Dat_cont.reshape(-1, 1))  # REENTRENAR MODELO
        scores = -IF.decision_function(Dat_cont.reshape(-1, 1))  # NUEVOS SCORES
        top_indices = np.argsort(scores)[-n_anom:]  # ANOMALÍAS MÁS DESTACADAS
        CR = scores[top_indices]  # EXTRAER SCORES DE INTERÉS
        R = np.mean(CR)  # CALCULAR TASA PROMEDIO DE AISLAMIENTO
        print(f"[INFO] Tasa promedio de aislamiento R={R:.4f}")

        # AJUSTAR D SEGÚN POSICIÓN DE R EN CUARTILES
        new_D = D
        if R < R_75 and D > D_min:
            new_D = max(int(D * (1 - betha)), D_min)  # REDUCIR PROFUNDIDAD
        elif R > R_25 and D < D_max:
            new_D = min(int(D * (1 + betha)), D_max)  # AUMENTAR PROFUNDIDAD

        # DETENER SI D SE ESTABILIZA
        if new_D == D:
            print(f"[INFO] Factor D ajustado encontrado: {D}")
            return D

        D = new_D  # ACTUALIZAR D

    # FINALIZAR SI SE ALCANZA EL LÍMITE DE ITERACIONES
    print(f"[WARN] Máximo número de iteraciones alcanzado, factor D final: {D}")
    return D


# [ MAIN MEJORADO ]
def ajustar_profundidad_maxima_mejorado(Dat, S, betha=0.03, random_state=None, max_iter=50, reps=5):
    np.random.seed(random_state)  # FIJAR SEMILLA ALEATORIA

    # [ CAMBIO ] CONTAMINAR DATASET CON 10% ANOMALIAS EN LUGAR DE 5%
    Dat_cont, indices_anom = contaminar_dat_5pct(Dat[:S], porcentaje=0.1, incremento=0.5, random_state=random_state)
    print(f"[INFO] Dataset contaminado con {len(indices_anom)} anomalías")

    # [ CAMBIO ] DEFINIR D_MIN Y D INICIAL COMO MEDIANA ENTRE MIN Y MAX
    D_max = int(log2(S))  # MAX PROFUNDIDAD
    D_min = max(1, int(D_max * 0.25))  # MIN PROFUNDIDAD
    D = max(D_min, int((D_min + D_max) / 2))  # INICIO EN MEDIANA
    print(f"[INFO] D inicial: {D}, D_min={D_min}, D_max={D_max}")

    # [ CAMBIO ] MÁS ÁRBOLES EN ISOLATION FOREST
    IF = IsolationForest(max_samples=S, max_features=1.0, n_estimators=150, random_state=random_state)  # ENTRENAR IF
    IF.fit(Dat_cont.reshape(-1, 1))
    scores = -IF.decision_function(Dat_cont.reshape(-1, 1))  # PUNTUACIONES
    n_anom = max(1, int(S * 0.05))  # NUMERO DE ANOMALIAS TOP
    top_indices = np.argsort(scores)[-n_anom:]
    CR = scores[top_indices]
    R_25, R_75 = np.percentile(CR, [25, 75])  # CUARTILES
    print(f"[INFO] R_25={R_25:.4f}, R_75={R_75:.4f}")

    iteration = 0
    last_R = None  # [ CAMBIO ] SEGUIMIENTO DE R ANTERIOR
    stable_count = 0  # [ CAMBIO ] CONTADOR DE ESTABILIDAD

    while iteration < max_iter:
        iteration += 1
        print(f"[INFO] Iteración {iteration}: Probando factor D={D}")
        max_samples = max(1, int(S * D / D_max))  # CALCULAR MAX_SAMPLES

        # [ CAMBIO ] PROMEDIO DE R SOBRE VARIAS REPETICIONES PARA REDUCIR VARIABILIDAD
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

        # AJUSTAR D
        new_D = D
        # [ CAMBIO ] USAR CEIL Y RESTRICCIÓN D_MIN+1 PARA NO REDUCIR DEMASIADO RÁPIDO
        if R < R_75 and D > D_min + 1:
            new_D = max(int(np.ceil(D * (1 - betha))), D_min)
        # [ CAMBIO ] AUMENTAR D MÁS FLEXIBLEMENTE
        elif R >= R_25 * 0.95 and D < D_max:
            new_D = min(int(np.ceil(D * (1 + betha))), D_max)

        # [ CAMBIO ] CRITERIO DE CONVERGENCIA USANDO last_R Y stable_count
        if last_R is not None and abs(R - last_R) < 0.005:
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
        last_R = R  # [ CAMBIO ] ACTUALIZAR last_R

    print(f"[WARN] Máximo número de iteraciones alcanzado, factor D final: {D}")
    return D


# LEER S DESDE JSON O USAR DEFAULT
if os.path.exists(HIP_JSON):
    with open(HIP_JSON, 'r', encoding='utf-8') as f:
        hip_data = json.load(f)
    S = hip_data.get('S', {}).get('value', 200)
else:
    S = 256  # DEFAULT


# [ HIPERPARÁMETROS ]
# DAT_NP → DATASET DE ENTRADA EN FORMATO NUMPY
# S → TAMAÑO DE MUESTRA POR ITERACIÓN
# BETHA → FACTOR DE AJUSTE DE PROFUNDIDAD
# RANDOM_STATE → SEMILLA PARA REPRODUCIBILIDAD
# MAX_ITER → NÚMERO MÁXIMO DE ITERACIONES

# D_ajustado = ajustar_profundidad_maxima(Dat_np, S=S, betha=0.2, random_state=42, max_iter=50)
# D_ajustado = ajustar_profundidad_maxima(Dat_np, S, betha=0.2, random_state=42) # Ajustado        
D_ajustado = ajustar_profundidad_maxima_mejorado( Dat_np, S=S, betha=0.5, random_state=42, max_iter=10) # 
print(f"[FIN] hiperparameters.json actualizado con D={D_ajustado}")


# CARGAR O CREAR JSON DE HIPERPARÁMETROS
if os.path.exists(HIP_JSON):
    with open(HIP_JSON, 'r', encoding='utf-8') as f:
        hip_data = json.load(f)  # LEER EXISTENTE
else:
    hip_data = {}  # CREAR NUEVO

# ACTUALIZAR O AÑADIR VALOR DE D
hip_data['D'] = {
    "value": D_ajustado,  # VALOR FINAL
    "description": "Factor controlling Isolation Forest effective depth",  # DESCRIPCION
    "adjustment_method": "Adjusted using average isolation rate R and quartiles",  # METODO AJUSTE
    "default": "log2(S) = 8 o 6"  # VALOR POR DEFECTO
}

# GUARDAR JSON ACTUALIZADO
with open(HIP_JSON, 'w', encoding='utf-8') as f:
    json.dump(hip_data, f, indent=4)  # ESCRIBIR JSON
