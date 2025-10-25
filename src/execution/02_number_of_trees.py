import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score
import json
import os

# PARÁMETROS DE ENTRADA Y SALIDA
INPUT_CSV = '../../results/preparation/05_variance_recortado.csv'  # CSV CON DATOS
HIP_JSON = '../../results/execution/hiperparameters.json'  # JSON DE HIPERPARÁMETROS

# CARGAR DATASET
df = pd.read_csv(INPUT_CSV)  # LEER CSV

# ELIMINAR COLUMNA 'is_anomaly' SI EXISTE
if 'is_anomaly' in df.columns:
    df = df.drop(columns=['is_anomaly'])  # ELIMINAR COLUMNA NO NECESARIA PARA AJUSTE

# CONVERTIR COLUMNAS NUMÉRICAS A NUMPY 2D
Dat_np = df.select_dtypes(include=[np.number]).values  # EXTRAER DATOS NUMÉRICOS

# CONTAMINA SUBCONJUNTO DEL DATASET CON ANOMALÍAS ARTIFICIALES
def contaminar_dat(Dat, S, porcentaje=0.25, incremento=0.5, random_state=None):
    np.random.seed(random_state)  # FIJAR SEMILLA PARA REPRODUCIBILIDAD
    indices_muestra = np.random.choice(Dat.shape[0], size=S, replace=False)  # SELECCIONAR MUESTRA
    muestra = Dat[indices_muestra, :].copy()  # COPIA DE SUBMUESTRA
    # n_anom = max(1, int(S * porcentaje))  # NÚMERO DE ANOMALÍAS A INSERTAR
    n_anom = max(3, int(S * porcentaje))  # [ CAMBIO ] al menos 3 anomalías
    indices_anom = np.random.choice(S, size=n_anom, replace=False)  # INDICES DE ANOMALÍAS
    col = np.random.randint(muestra.shape[1])  # COLUMNA A CONTAMINAR
    muestra[indices_anom, col] *= (1 + incremento)  # APLICAR ANOMALÍAS
    return muestra, indices_anom, indices_muestra  # RETORNAR SUBMUESTRA Y ANOMALÍAS


# [ MAIN ]
def ajustar_numero_arboles(Dat, S, T_min=5, T_max=100, step=5, N=3, F1sta=0.01, random_state=None):
    np.random.seed(random_state)  # FIJAR SEMILLA

    # [ CAMBIO ] aumentar porcentaje de anomalías y mínimo de anomalías
    Dat_cont, indices_anom_real, _ = contaminar_dat(
        Dat, S, porcentaje=0.20, incremento=0.5, random_state=random_state
    )
    print(f"[INFO] Dataset contaminado con {len(indices_anom_real)} anomalías artificiales")

    T = T_min
    F1_list = []

    # BÚSQUEDA ITERATIVA DEL NÚMERO DE ÁRBOLES
    while T <= T_max:
        print(f"[INFO] Probando T={T} árboles")
        IF = IsolationForest(n_estimators=T, contamination=0.01, random_state=random_state)

        # [ CAMBIO ] promediar F1-score sobre varias repeticiones
        reps = 5
        F1_sum = 0
        for _ in range(reps):
            IF.fit(Dat_cont)  # ENTRENAR ISOLATION FOREST
            scores = -IF.decision_function(Dat_cont)  # CALCULAR SCORE DE ANOMALÍA
            n_anom_pred = max(3, int(S * 0.20))  # [ CAMBIO ] mínimo 3 anomalías predichas
            indices_pred = np.argsort(scores)[-n_anom_pred:]

            y_true = np.zeros(S)
            y_true[indices_anom_real] = 1  # ANOMALÍAS REALES
            y_pred = np.zeros(S)
            y_pred[indices_pred] = 1  # ANOMALÍAS PREDICHAS
            F1_sum += f1_score(y_true, y_pred)

        F1 = F1_sum / reps  # [ CAMBIO ] promedio
        F1_list.append(F1)
        print(f"[INFO] F1-score: {F1:.4f}")

        # VERIFICAR ESTABILIDAD DEL F1-SCORE EN ÚLTIMAS N ITERACIONES
        if len(F1_list) >= N:
            cumple = sum(1 for f in F1_list[-N:] if f >= F1sta)  # [ CAMBIO ] >= en vez de <=
            if cumple == N:
                print(f"[INFO] F1-score estable detectado. T final: {T}")
                return T  # RETORNAR T FINAL AJUSTADO

        T += step  # AUMENTAR NÚMERO DE ÁRBOLES

    print(f"[INFO] F1-score no estabilizó. T final = {T_max}")
    return T_max  # RETORNAR T MÁXIMO SI NO SE ESTABILIZA


# [ MAIN MEJORADO ]
def ajustar_numero_arboles_mejorado(Dat, S, T_min=5, T_max=100, step=1, N=5, delta=0.01, random_state=None):
    np.random.seed(random_state)  # FIJAR SEMILLA GLOBAL (igual que versión anterior)

    T = T_min  # INICIALIZAR número de árboles
    F1_list = []  # LISTA PARA GUARDAR F1 promedio por T

    while T <= T_max:
        print(f"[INFO] Probando T={T} árboles")

        # [ CAMBIO ] respecto a versión anterior:
        # Antes: contamination=0.01, ahora se aumenta a 0.25 para mejorar entrenamiento con pocas anomalías
        IF = IsolationForest(n_estimators=T, contamination=0.25, random_state=random_state)

        reps = 5  # [ CAMBIO ] promediar F1 sobre varias repeticiones (antes solo 1) para reducir ruido
        F1_sum = 0
        for r in range(reps):
            # [ CAMBIO ] aumentar porcentaje de anomalías al 25% y usar semilla distinta por repetición
            # Antes: porcentaje 20% y misma semilla -> submuestras idénticas
            Dat_cont, indices_anom_real, _ = contaminar_dat(
                Dat, S, porcentaje=0.25, incremento=0.5, random_state=random_state+r
            )

            IF.fit(Dat_cont)  # ENTRENAR Isolation Forest
            scores = -IF.decision_function(Dat_cont)  # CALCULAR scores

            # [ CAMBIO ] asegurar mínimo de 3 anomalías predichas
            # Antes podía predecirse solo 1, causando F1=0 si no coincidía
            n_anom_pred = max(3, int(S * 0.25))
            indices_pred = np.argsort(scores)[-n_anom_pred:]

            y_true = np.zeros(S)
            y_true[indices_anom_real] = 1  # VECTOR DE ANOMALÍAS REALES
            y_pred = np.zeros(S)
            y_pred[indices_pred] = 1  # VECTOR DE ANOMALÍAS PREDICHAS

            F1_sum += f1_score(y_true, y_pred)  # ACUMULAR F1 por repetición

        F1 = F1_sum / reps  # [ CAMBIO ] calcular promedio F1 (antes se tomaba solo 1 F1)
        F1_list.append(F1)
        print(f"[INFO] F1-score promedio: {F1:.4f}")

        # [ CAMBIO ] estabilidad ahora definida como variación máxima < delta en últimas N iteraciones
        # Antes: se verificaba si F1 ≥ F1sta en N iteraciones consecutivas
        if len(F1_list) >= N:
            max_diff = max(F1_list[-N:]) - min(F1_list[-N:])
            if max_diff < delta:
                print(f"[INFO] F1-score estable detectado. T final: {T}")
                return T  # RETOR


# LEER S DESDE JSON
if os.path.exists(HIP_JSON):
    with open(HIP_JSON, 'r', encoding='utf-8') as f:
        hip_data = json.load(f)
    S = hip_data.get('S', {}).get('value', 200)  # Si no existe, usar 200 como default
else:
    S = 256  # Default si no existe JSON


# CALL
# T_ajustado = ajustar_numero_arboles(Dat_np, S, T_min=50, T_max=100, step=5, N=3, F1sta=0.01, random_state=42) # Original
T_ajustado = ajustar_numero_arboles(Dat_np, S, T_min=50, T_max=100, step=1, N=10, F1sta=0.001, random_state=42) # Ajustado
# T_ajustado = ajustar_numero_arboles_mejorado(Dat_np, S, T_min=50, T_max=100, step=1, N=3, delta=0.001, random_state=42) # Propuesto
print(f"[FIN] hiperparameters.json actualizado con T={T_ajustado}")


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