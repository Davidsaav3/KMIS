import numpy as np 
import pandas as pd
from sklearn.ensemble import IsolationForest
from math import log2
import json
import os

# PARÁMETROS DE ENTRADA Y SALIDA
INPUT_CSV = '../../results/execution/00_contaminated.csv'  # RUTA DEL CSV DE DATOS
HIP_JSON = '../../results/execution/hiperparameters.json'  # RUTA DEL JSON DE HIPERPARÁMETROS

# CARGAR DATASET
df = pd.read_csv(INPUT_CSV)  # LEER CSV DESDE DISCO

# ELIMINAR COLUMNA 'is_anomaly' SI EXISTE
if 'is_anomaly' in df.columns:
    df = df.drop(columns=['is_anomaly'])  # ELIMINAR COLUMNA NO RELEVANTE

# CONVERTIR DATOS NUMÉRICOS A VECTOR 1D
Dat_np = df.select_dtypes(include=[np.number]).values.flatten()  # EXTRAER Y APLANAR DATOS NUMÉRICOS

# FUNCIÓN PARA CONTAMINAR DATOS CON ANOMALÍAS ARTIFICIALES
def contaminar_dat_5pct(Dat, porcentaje=0.05, incremento=0.5, random_state=None):
    """CONTAMINA 5% DE LOS DATOS AUMENTANDO SUS VALORES"""
    np.random.seed(random_state)  # FIJAR SEMILLA
    S = len(Dat)  # NÚMERO TOTAL DE ELEMENTOS
    n_anom = max(1, int(S * porcentaje))  # CANTIDAD DE ANOMALÍAS A INTRODUCIR
    Dat_contaminado = Dat.copy()  # COPIAR DATOS ORIGINALES
    indices_anom = np.random.choice(S, size=n_anom, replace=False)  # ÍNDICES ALEATORIOS A ALTERAR
    Dat_contaminado[indices_anom] *= (1 + incremento)  # AUMENTAR VALORES SELECCIONADOS
    return Dat_contaminado, indices_anom  # RETORNAR DATOS CONTAMINADOS Y SUS ÍNDICES

# FUNCIÓN PARA AJUSTAR EL FACTOR D (PROFUNDIDAD EFECTIVA)
def ajustar_profundidad_maxima(Dat, S, β=0.2, random_state=None, max_iter=50):
    """AJUSTA D SEGÚN LA TASA PROMEDIO DE AISLAMIENTO"""
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
            new_D = max(int(D * (1 - β)), D_min)  # REDUCIR PROFUNDIDAD
        elif R > R_25 and D < D_max:
            new_D = min(int(D * (1 + β)), D_max)  # AUMENTAR PROFUNDIDAD

        # DETENER SI D SE ESTABILIZA
        if new_D == D:
            print(f"[INFO] Factor D ajustado encontrado: {D}")
            return D

        D = new_D  # ACTUALIZAR D

    # FINALIZAR SI SE ALCANZA EL LÍMITE DE ITERACIONES
    print(f"[WARN] Máximo número de iteraciones alcanzado, factor D final: {D}")
    return D

# EJECUTAR PROCESO DE AJUSTE DE D
S = 256  # TAMAÑO DE SUBMUESTRA
D_ajustado = ajustar_profundidad_maxima(Dat_np, S, β=0.2, random_state=42)
print(f"[INFO] Factor D final ajustado: {D_ajustado}")

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
    "default": "log2(S)"  # VALOR POR DEFECTO
}

# GUARDAR JSON ACTUALIZADO
with open(HIP_JSON, 'w', encoding='utf-8') as f:
    json.dump(hip_data, f, indent=4)  # ESCRIBIR JSON FORMATEADO

print(f"[INFO] hiperparameters.json actualizado con D={D_ajustado}")
