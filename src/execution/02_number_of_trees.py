import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score
import json
import os

# PARÁMETROS DE ENTRADA Y SALIDA
INPUT_CSV = '../../results/execution/00_contaminated.csv'  # CSV CON DATOS
HIP_JSON = '../../results/execution/hiperparameters.json'  # JSON DE HIPERPARÁMETROS

# CARGAR DATASET
df = pd.read_csv(INPUT_CSV)  # LEER CSV

# ELIMINAR COLUMNA 'is_anomaly' SI EXISTE
if 'is_anomaly' in df.columns:
    df = df.drop(columns=['is_anomaly'])  # ELIMINAR COLUMNA NO NECESARIA PARA AJUSTE

# CONVERTIR COLUMNAS NUMÉRICAS A NUMPY 2D
Dat_np = df.select_dtypes(include=[np.number]).values  # EXTRAER DATOS NUMÉRICOS

# FUNCIONES
def contaminar_dat(Dat, S, porcentaje=0.01, incremento=0.5, random_state=None):
    """CONTAMINA SUBCONJUNTO DEL DATASET CON ANOMALÍAS ARTIFICIALES"""
    np.random.seed(random_state)  # FIJAR SEMILLA PARA REPRODUCIBILIDAD
    indices_muestra = np.random.choice(Dat.shape[0], size=S, replace=False)  # SELECCIONAR MUESTRA
    muestra = Dat[indices_muestra, :].copy()  # COPIA DE SUBMUESTRA
    n_anom = max(1, int(S * porcentaje))  # NÚMERO DE ANOMALÍAS A INSERTAR
    indices_anom = np.random.choice(S, size=n_anom, replace=False)  # INDICES DE ANOMALÍAS
    col = np.random.randint(muestra.shape[1])  # COLUMNA A CONTAMINAR
    muestra[indices_anom, col] *= (1 + incremento)  # APLICAR ANOMALÍAS
    return muestra, indices_anom, indices_muestra  # RETORNAR SUBMUESTRA Y ANOMALÍAS

def ajustar_numero_arboles(Dat, S, T_min=5, T_max=100, step=5, N=3, F1sta=0.01, random_state=None):
    """AJUSTA NÚMERO DE ÁRBOLES SEGÚN ESTABILIDAD DEL F1-SCORE"""
    np.random.seed(random_state)  # FIJAR SEMILLA
    # CONTAMINAR SUBMUESTRA
    Dat_cont, indices_anom_real, _ = contaminar_dat(Dat, S, porcentaje=0.01, incremento=0.5, random_state=random_state)
    print(f"[INFO] Dataset contaminado con {len(indices_anom_real)} anomalías artificiales")
    T = T_min
    F1_list = []

    # BÚSQUEDA ITERATIVA DEL NÚMERO DE ÁRBOLES
    while T <= T_max:
        print(f"[INFO] Probando T={T} árboles")
        IF = IsolationForest(n_estimators=T, contamination=0.01, random_state=random_state)
        IF.fit(Dat_cont)  # ENTRENAR ISOLATION FOREST CON SUBMUESTRA
        scores = -IF.decision_function(Dat_cont)  # CALCULAR SCORE DE ANOMALÍA
        n_anom_pred = max(1, int(S * 0.01))  # NÚMERO DE ANOMALÍAS A PREDECIR
        indices_pred = np.argsort(scores)[-n_anom_pred:]  # INDICES MÁS ANÓMALOS

        # CALCULAR F1-SCORE
        y_true = np.zeros(S)
        y_true[indices_anom_real] = 1  # ANOMALÍAS REALES
        y_pred = np.zeros(S)
        y_pred[indices_pred] = 1  # ANOMALÍAS PREDICHAS
        F1 = f1_score(y_true, y_pred)  # F1 SCORE
        F1_list.append(F1)
        print(f"[INFO] F1-score: {F1:.4f}")

        # VERIFICAR ESTABILIDAD DEL F1-SCORE EN ÚLTIMAS N ITERACIONES
        if len(F1_list) >= N:
            cumple = sum(1 for f in F1_list[-N:] if f <= F1sta)
            if cumple == N:
                print(f"[INFO] F1-score estable detectado. T final: {T}")
                return T  # RETORNAR T FINAL AJUSTADO

        T += step  # AUMENTAR NÚMERO DE ÁRBOLES PARA PRÓXIMA ITERACIÓN

    print(f"[INFO] F1-score no estabilizó. T final = {T_max}")
    return T_max  # RETORNAR T MÁXIMO SI NO SE ESTABILIZA

# EJECUCIÓN DEL AJUSTE
S = 200  # TAMAÑO DE MUESTRA PARA AJUSTE
T_ajustado = ajustar_numero_arboles(Dat_np, S, T_min=5, T_max=100, step=5, N=3, F1sta=0.01, random_state=42)
print(f"[INFO] Número de árboles final ajustado: {T_ajustado}")

# CREAR O ACTUALIZAR JSON DE HIPERPARÁMETROS
if os.path.exists(HIP_JSON):
    with open(HIP_JSON, 'r', encoding='utf-8') as f:
        hip_data = json.load(f)  # CARGAR JSON EXISTENTE
else:
    hip_data = {}  # CREAR NUEVO JSON

# GUARDAR VALOR AJUSTADO DE T
hip_data['T'] = {
    "value": T_ajustado,  # VALOR AJUSTADO
    "description": "Number of trees in the Isolation Forest",  # DESCRIPCIÓN
    "adjustment_method": "F1-score stabilization over iterations",  # MÉTODO DE AJUSTE
    "default": 100  # VALOR POR DEFECTO
}

with open(HIP_JSON, 'w', encoding='utf-8') as f:
    json.dump(hip_data, f, indent=4)  # GUARDAR JSON

print(f"[INFO] hiperparameters.json actualizado con T={T_ajustado}")
