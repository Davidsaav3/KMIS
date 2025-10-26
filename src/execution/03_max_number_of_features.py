import numpy as np 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import json
import os

# PARÁMETROS DE ENTRADA Y SALIDA
INPUT_CSV = '../../results/preparation/05_variance_recortado.csv'  # CSV CON DATASET
HIP_JSON = '../../results/execution/hiperparameters.json'  # JSON DE HIPERPARÁMETROS

# CARGAR DATASET
df = pd.read_csv(INPUT_CSV)  # LEER CSV

# ELIMINAR COLUMNA 'is_anomaly' SI EXISTE
if 'is_anomaly' in df.columns:
    df = df.drop(columns=['is_anomaly'])  # ELIMINAR COLUMNA NO NECESARIA

# CONVERTIR COLUMNAS NUMÉRICAS A NUMPY
Dat_np = df.select_dtypes(include=[np.number]).values  # EXTRAER SOLO DATOS NUMÉRICOS


# [ MAIN ]
def ajustar_numero_caracteristicas(Dat, F_inicial=1.0, alpha_reduccion=0.5, alpha_aumento=1.5, random_state=None):
    # AJUSTA EL PARÁMETRO F SEGÚN LA VARIANZA MEDIA DE SUBMUESTRAS
    np.random.seed(random_state)  # FIJAR SEMILLA PARA REPRODUCIBILIDAD
    scaler = MinMaxScaler()
    Dat_norm = scaler.fit_transform(Dat)  # NORMALIZAR DATOS A [0,1]
    print("[INFO] Dataset normalizado a rango [0,1]")

    num_features = Dat.shape[1]  # NÚMERO TOTAL DE COLUMNAS
    F_min = 1 / num_features  # LÍMITE INFERIOR DE F
    F_max = 1.0  # LÍMITE SUPERIOR DE F
    F = F_inicial  # VALOR INICIAL DE F
    print(f"[INFO] F inicial: {F:.4f}")

    # BUCLE DE AJUSTE ITERATIVO DE F
    while True:
        n_selected = max(1, int(F * num_features))  # NÚMERO DE CARACTERÍSTICAS A USAR
        selected_idx = np.random.choice(num_features, size=n_selected, replace=False)  # SELECCIÓN ALEATORIA DE COLUMNAS
        selected_features = Dat_norm[:, selected_idx]  # EXTRAER SUBMUESTRA

        sigma_2 = np.var(selected_features, axis=0)  # CALCULAR VARIANZA POR COLUMNA
        V_prom = np.mean(sigma_2)  # VARIANZA PROMEDIO
        Q1, Q2, Q3, Q4 = np.percentile(sigma_2, [25, 50, 75, 100])  # CALCULAR CUARTILES
        BQ1F = (np.max([Q1]) + np.min([Q2])) / 2  # LÍMITE INFERIOR DE REFERENCIA
        BQ4F = (np.max([Q3]) + np.min([Q4])) / 2  # LÍMITE SUPERIOR DE REFERENCIA

        print(f"[INFO] V_prom={V_prom:.4f}, BQ1F={BQ1F:.4f}, BQ4F={BQ4F:.4f}, F={F:.4f}")

        # AJUSTE HACIA ABAJO SI VARIANZA ES DEMASIADO ALTA
        if V_prom > BQ4F:
            if F * alpha_reduccion < F_min:  # NO BAJAR DE LÍMITE
                print("[INFO] F alcanzó F_min, se retorna F")
                return F
            F *= alpha_reduccion  # REDUCIR F
            print(f"[INFO] Reducción de F a {F:.4f}")

        # AJUSTE HACIA ARRIBA SI VARIANZA ES DEMASIADO BAJA
        elif V_prom < BQ1F:
            if F * alpha_aumento > F_max:  # NO SUPERAR LÍMITE
                print("[INFO] F alcanzó F_max, se retorna F")
                return F
            F *= alpha_aumento  # AUMENTAR F
            print(f"[INFO] Aumento de F a {F:.4f}")

        # CONDICIÓN DE ESTABILIDAD ALCANZADA
        else:
            print("[INFO] F ajustado encontrado")
            return F


# [ MAIN MEJORADO ]
def ajustar_numero_caracteristicas_mejorado(Dat, F_inicial=1.0, alpha_reduccion=0.9, alpha_aumento=1.05, reps=5, F_min=0.1, random_state=None):
    np.random.seed(random_state)  # FIJAR SEMILLA
    scaler = MinMaxScaler()
    Dat_norm = scaler.fit_transform(Dat)  # NORMALIZAR [0,1]
    print("[INFO] Dataset normalizado a rango [0,1]")

    num_features = Dat.shape[1]  # NÚMERO DE COLUMNAS
    F_max = 1.0  # LÍMITE SUPERIOR
    F_min = max(F_min, 1.0 / num_features)  # [CAMBIO] MÍNIMO ABSOLUTO
    F = min(F_inicial, F_max)  # [CAMBIO] NO SUPERAR F_MAX AL INICIO
    print(f"[INFO] F inicial: {F:.4f}")

    max_iter = 50  # [CAMBIO] LÍMITE ITERACIONES
    iter_count = 0  # CONTADOR

    while iter_count < max_iter:
        iter_count += 1
        V_prom_list = []  # [CAMBIO] LISTA PARA PROMEDIO VARIANZAS

        for r in range(reps):  # [CAMBIO] VARIAS SUBMUESTRAS
            n_selected = max(1, int(F * num_features))
            selected_idx = np.random.choice(num_features, size=n_selected, replace=False)
            selected_features = Dat_norm[:, selected_idx]
            V_prom_list.append(np.mean(np.var(selected_features, axis=0)))  # [CAMBIO] PROMEDIO SUBMUESTRA

        V_prom = np.mean(V_prom_list)  # [CAMBIO] PROMEDIO REPETICIONES
        sigma_2_full = np.var(Dat_norm, axis=0)
        Q1, Q2, Q3, Q4 = np.percentile(sigma_2_full, [25, 50, 75, 100])
        BQ1F = (Q1 + Q2) / 2
        BQ4F = (Q3 + Q4) / 2

        print(f"[INFO] Iter {iter_count}: V_prom={V_prom:.4f}, BQ1F={BQ1F:.4f}, BQ4F={BQ4F:.4f}, F={F:.4f}")

        # REDUCCIÓN AGRESIVA
        F_new = max(F * alpha_reduccion, F_min)
        F_new = min(F_new, F_max)
        if F_new == F:
            print("[INFO] F alcanzó F_min o F_max, se retorna F")
            break
        F = F_new
        print(f"[INFO] Reducción agresiva de F a {F:.4f}")

        # AUMENTO LIGERO SI VARIANZA BAJA
        if V_prom < BQ1F:
            F = min(F * alpha_aumento, F_max)
            print(f"[INFO] Ajuste aumento F a {F:.4f} por varianza baja")

        # CONDICIÓN DE PARADA
        if F <= F_min + 0.01:
            print("[INFO] F cerca de F_min, finalizando ajuste")
            break

    print("[INFO] F ajustado encontrado")
    return F


# [ HIPERPARÁMETROS ]
# DAT_NP → DATASET EN NUMPY
# F_INICIAL → VALOR INICIAL DE CARACTERÍSTICAS
# ALPHA_REDUCCION → FACTOR DE REDUCCIÓN
# ALPHA_AUMENTO → FACTOR DE AUMENTO
# REPS → REPETICIONES POR ITERACIÓN
# F_MIN → MÍNIMO DE CARACTERÍSTICAS
# RANDOM_STATE → SEMILLA ALEATORIA

# F_ajustado = ajustar_numero_caracteristicas(Dat_np, F_inicial=1.0, alpha_reduccion=0.5, alpha_aumento=1.5, random_state=42) # Original
# F_ajustado = ajustar_numero_caracteristicas_mejorado(Dat_np, F_inicial=1.0, alpha_reduccion=0.5, alpha_aumento=1.5, reps=5, random_state=42) # Ajustado
# F_ajustado = ajustar_numero_caracteristicas_mejorado(Dat, F_inicial=1.0, alpha_reduccion=0.9, alpha_aumento=1.05, reps=5, F_min=0.1, random_state=None)
F_ajustado = ajustar_numero_caracteristicas_mejorado(Dat_np, F_inicial=1.0, alpha_reduccion=0.98, alpha_aumento=1.3, reps=10, F_min=0.25, random_state=42)
print(f"[FIN] hiperparameters.json actualizado con F={F_ajustado:.4f}")


# CREAR O ACTUALIZAR JSON
if os.path.exists(HIP_JSON):
    with open(HIP_JSON, 'r', encoding='utf-8') as f:
        hip_data = json.load(f)  # LEER JSON EXISTENTE
else:
    hip_data = {}  # NUEVO JSON

# GUARDAR VALOR AJUSTADO DE F
hip_data['F'] = {
    "value": F_ajustado,  # VALOR FINAL
    "description": "Maximum number of features considered per tree",  # DESCRIPCIÓN
    "adjustment_method": "Increment/decrement search based on variance",  # MÉTODO AJUSTE
    "default": 1.0  # VALOR POR DEFECTO
}

# GUARDAR JSON
with open(HIP_JSON, 'w', encoding='utf-8') as f:
    json.dump(hip_data, f, indent=4)  # GUARDAR CON FORMATO
