import pandas as pd
import numpy as np
import json
import os

# PARÁMETROS DE ENTRADA Y SALIDA
INPUT_CSV = '../../results/execution/00_contaminated.csv'  # RUTA DEL CSV DE DATOS
HIP_JSON = '../../results/execution/hiperparameters.json'  # RUTA DEL JSON DE HIPERPARÁMETROS

# CARGAR DATASET
df = pd.read_csv(INPUT_CSV)  # LEER CSV CON DATOS

# ELIMINAR COLUMNA 'is_anomaly' SI EXISTE
if 'is_anomaly' in df.columns:
    df = df.drop(columns=['is_anomaly'])  # ELIMINAR COLUMNA NO NECESARIA PARA AJUSTE

# APLANAR COLUMNAS NUMÉRICAS A VECTORIZADO 1D
data = df.select_dtypes(include=[np.number]).values.flatten()  # EXTRAER Y APLANAR DATOS NUMÉRICOS

# FUNCIONES
def desviacion_tipica(data):
    """CALCULA LA DESVIACIÓN TÍPICA DE UN VECTOR 1D"""
    return np.std(data)  # DEVUELVE DESVIACIÓN ESTÁNDAR

def ajustar_tamano_muestra(Dat, S_inicial=256, e_sigma=0.05, IncDat=0.1, random_state=None):
    """AJUSTA EL TAMAÑO DE MUESTRA HASTA IGUALAR LA DESVIACIÓN TÍPICA DEL CONJUNTO COMPLETO"""
    np.random.seed(random_state)  # FIJAR SEMILLA PARA REPRODUCIBILIDAD
    S = S_inicial
    print(f"[INFO] Tamaño inicial de muestra: {S}")

    sigma_o = desviacion_tipica(Dat)  # DESVIACIÓN TÍPICA DEL CONJUNTO ORIGINAL
    print(f"[INFO] Desviación típica del conjunto original sigma_o: {sigma_o:.4f}")

    # CALCULAR DESVIACIÓN DE LA MUESTRA INICIAL
    sigma_min = desviacion_tipica(np.random.choice(Dat, size=min(S, len(Dat)), replace=False))
    print(f"[INFO] Desviación típica de la muestra inicial sigma_min: {sigma_min:.4f}")

    # INCREMENTAR TAMAÑO DE MUESTRA HASTA ALCANZAR DESVIACIÓN OBJETIVO
    while S < len(Dat) and not (sigma_o - e_sigma <= sigma_min <= sigma_o + e_sigma):
        S = int(min(S * (1 + IncDat), len(Dat)))  # AUMENTAR MUESTRA
        print(f"[INFO] Incrementando tamaño de muestra a: {S}")

        muestra = np.random.choice(Dat, size=S, replace=False)  # NUEVA MUESTRA ALEATORIA
        sigma_min = desviacion_tipica(muestra)  # CALCULAR DESVIACIÓN DE LA MUESTRA
        print(f"[INFO] Desviación típica de la nueva muestra sigma_min: {sigma_min:.4f}")

    print(f"[INFO] Tamaño de muestra final ajustado: {S}")
    return S  # RETORNAR TAMAÑO DE MUESTRA AJUSTADO

# AJUSTAR TAMAÑO DE MUESTRA
S_ajustado = ajustar_tamano_muestra(data, S_inicial=256, e_sigma=0.05, IncDat=0.1, random_state=42)

# LEER O CREAR JSON DE HIPERPARÁMETROS
if os.path.exists(HIP_JSON):
    with open(HIP_JSON, 'r', encoding='utf-8') as f:
        hip_data = json.load(f)  # CARGAR JSON EXISTENTE
else:
    hip_data = {}  # CREAR NUEVO JSON SI NO EXISTE

# ACTUALIZAR INFORMACIÓN DE S
hip_data['S'] = {
    "value": S_ajustado,  # VALOR AJUSTADO
    "description": "Sample size for Isolation Forest",  # DESCRIPCIÓN
    "adjustment_method": "Standard deviation based incremental sampling",  # MÉTODO DE AJUSTE
    "default": 256  # VALOR POR DEFECTO
}

# GUARDAR JSON ACTUALIZADO
with open(HIP_JSON, 'w', encoding='utf-8') as f:
    json.dump(hip_data, f, indent=4)  # GUARDAR HIPERPARÁMETROS

print(f"[INFO] hiperparameters.json actualizado con S={S_ajustado}")
