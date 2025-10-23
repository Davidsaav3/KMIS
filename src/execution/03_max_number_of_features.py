import numpy as np 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import json
import os

# PARÁMETROS DE ENTRADA Y SALIDA
INPUT_CSV = '../../results/execution/00_contaminated.csv'  # RUTA DEL CSV CON DATASET
HIP_JSON = '../../results/execution/hiperparameters.json'  # RUTA DEL JSON DE HIPERPARÁMETROS

# CARGAR DATASET
df = pd.read_csv(INPUT_CSV)  # LEER CSV DE ENTRADA

# ELIMINAR COLUMNA 'is_anomaly' SI EXISTE
if 'is_anomaly' in df.columns:
    df = df.drop(columns=['is_anomaly'])  # ELIMINAR COLUMNA NO NECESARIA

# CONVERTIR COLUMNAS NUMÉRICAS A ARRAY NUMPY
Dat_np = df.select_dtypes(include=[np.number]).values  # EXTRAER SOLO DATOS NUMÉRICOS


# FUNCIÓN PARA AJUSTAR F (NÚMERO DE CARACTERÍSTICAS)
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


# MAIN
F_ajustado = ajustar_numero_caracteristicas(Dat_np, F_inicial=1.0, alpha_reduccion=0.5, alpha_aumento=1.5, random_state=42)
print(f"[INFO] Número máximo de características final ajustado: {F_ajustado:.4f}")


# ACTUALIZAR O CREAR JSON DE HIPERPARÁMETROS
if os.path.exists(HIP_JSON):
    with open(HIP_JSON, 'r', encoding='utf-8') as f:
        hip_data = json.load(f)  # LEER JSON EXISTENTE
else:
    hip_data = {}  # CREAR NUEVO SI NO EXISTE
# GUARDAR NUEVO VALOR DE F EN JSON
hip_data['F'] = {
    "value": F_ajustado,  # VALOR FINAL AJUSTADO
    "description": "Maximum number of features considered per tree",  # DESCRIPCIÓN
    "adjustment_method": "Increment/decrement search based on variance",  # MÉTODO DE AJUSTE
    "default": 1.0  # VALOR POR DEFECTO
}
# ESCRIBIR JSON ACTUALIZADO
with open(HIP_JSON, 'w', encoding='utf-8') as f:
    json.dump(hip_data, f, indent=4)  # GUARDAR CON FORMATO

print(f"[FIN] hiperparameters.json actualizado con F={F_ajustado:.4f}")
