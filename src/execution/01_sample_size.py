import pandas as pd
import numpy as np
import json
import os

# PARÁMETROS DE ENTRADA Y SALIDA
INPUT_CSV = '../../results/execution/00_contaminated.csv'  # RUTA CSV DATOS
HIP_JSON = '../../results/execution/hiperparameters.json'  # RUTA JSON HIPERPARÁMETROS

# CARGAR DATASET
df = pd.read_csv(INPUT_CSV)  # LEER CSV

# ELIMINAR COLUMNA 'is_anomaly' SI EXISTE
if 'is_anomaly' in df.columns:
    df = df.drop(columns=['is_anomaly'])  # ELIMINAR COLUMNA NO NECESARIA

# APLANAR COLUMNAS NUMÉRICAS A VECTORIZADO 1D
data = df.select_dtypes(include=[np.number]).values.flatten()  # EXTRAER Y APLANAR DATOS


# [ MAIN ]
def ajustar_tamano_muestra(Dat, S_inicial=256, e_sigma=0.05, IncDat=0.1, random_state=None):
    np.random.seed(random_state)  # FIJAR SEMILLA PARA REPRODUCIBILIDAD
    S = S_inicial
    print(f"[INFO] Tamaño inicial de muestra: {S}")

    sigma_o = np.std(Dat)  # DESVIACIÓN ESTANDAR DEL CONJUNTO ORIGINAL
    print(f"[INFO] Desviación ESTANDAR del conjunto original sigma_o: {sigma_o:.4f}")

    # CALCULAR DESVIACIÓN DE LA MUESTRA INICIAL
    sigma_min = np.std(np.random.choice(Dat, size=min(S, len(Dat)), replace=False))
    print(f"[INFO] Desviación ESTANDAR de la muestra inicial sigma_min: {sigma_min:.4f}")

    # INCREMENTAR TAMAÑO DE MUESTRA HASTA ALCANZAR DESVIACIÓN OBJETIVO
    while S < len(Dat) and not (sigma_o - e_sigma <= sigma_min <= sigma_o + e_sigma):
        S = int(min(S * (1 + IncDat), len(Dat)))  # AUMENTAR MUESTRA
        print(f"[INFO] Incrementando tamaño de muestra a: {S}")

        muestra = np.random.choice(Dat, size=S, replace=False)  # NUEVA MUESTRA ALEATORIA
        sigma_min = np.std(muestra)  # CALCULAR DESVIACIÓN DE LA MUESTRA
        print(f"[INFO] Desviación ESTANDAR de la nueva muestra sigma_min: {sigma_min:.4f}")

    print(f"[INFO] Tamaño de muestra final ajustado: {S}")
    return S  # RETORNAR TAMAÑO DE MUESTRA AJUSTADO


# [ MAIN MEJORADO ]
def ajustar_tamano_muestra_mejorado(Dat, S_inicial=40, e_sigma=0.01, IncDat=0.05, reps=10, random_state=None):
    np.random.seed(random_state)  # FIJAR SEMILLA
    S = S_inicial  # INICIALIZAR TAMAÑO
    print(f"[INFO] TAMAÑO INICIAL MUESTRA: {S}")

    sigma_obj = np.std(Dat)  # DESVIACIÓN ESTANDAR CONJUNTO COMPLETO
    # [ CAMBIO ] ANTES ERA sigma_o, no se usaba promedio submuestras.

    # [ CAMBIO ] FUNCIÓN INTERNA PARA DESVIACIÓN PROMEDIO DE SUBMUESTRAS
    def sigma_promedio(S):
        muestras = [np.random.choice(Dat, size=S, replace=False) for _ in range(reps)]  # [ CAMBIO ] VARIAS SUBMUESTRAS
        return np.mean([np.std(m) for m in muestras])  # [ CAMBIO ] PROMEDIO DESVIACIONES

    sigma_muestra = sigma_promedio(S)  # [ CAMBIO ] DESVIACIÓN INICIAL PROMEDIO
    print(f"[INFO] DESVIACIÓN PROMEDIO MUESTRA INICIAL: {sigma_muestra:.4f}")

    # [ CAMBIO ] BUCLE OPTIMIZADO: INCREMENTO DINÁMICO SEGÚN DIFERENCIA SIGMA
    while S < len(Dat) and not (sigma_obj - e_sigma <= sigma_muestra <= sigma_obj + e_sigma):
        incremento = max(1, int(S * IncDat * abs(sigma_obj - sigma_muestra) / sigma_obj))  
        # [ CAMBIO ] INCREMENTO VARIABLE SEGÚN ERROR RELATIVO

        S = min(S + incremento, len(Dat))  # ACTUALIZAR TAMAÑO SIN SUPERAR TOTAL
        sigma_muestra = sigma_promedio(S)  # [ CAMBIO ] RECALCULAR PROMEDIO

        print(f"[INFO] INCREMENTANDO TAMAÑO MUESTRA A: {S}")  # INFORMAR NUEVO TAMAÑO
        print(f"[INFO] DESVIACIÓN PROMEDIO NUEVA MUESTRA: {sigma_muestra:.4f}")  # INFORMAR NUEVA DESVIACIÓN

    print(f"[INFO] TAMAÑO FINAL MUESTRA AJUSTADO: {S}")  # MOSTRAR RESULTADO FINAL
    return S  # RETORNAR TAMAÑO AJUSTADO


# [ HIPERPARÁMETROS ]
# DATA → DATASET BASE PARA CALCULAR TAMAÑO ÓPTIMO MUESTRA
# S_INICIAL → TAMAÑO INICIAL MUESTRA, PUNTO DE PARTIDA AJUSTE
# E_SIGMA → ERROR/TOLERANCIA CONVERGENCIA (MENOR = MÁS PRECISIÓN)
# INCDAT → INCREMENTO PORCENTAJE PARA AJUSTE
# REPS → NÚMERO REPETICIONES POR ITERACIÓN PARA ESTABILIDAD
# RANDOM_STATE → SEMILLA ALEATORIA PARA REPRODUCIBILIDAD

# S_ajustado = ajustar_tamano_muestra(data, S_inicial=256, e_sigma=0.05, IncDat=0.1, random_state=42)  # ORIGINAL
# S_ajustado = ajustar_tamano_muestra(data, S_inicial=40, e_sigma=0.01, IncDat=0.05, random_state=42)  # AJUSTADO
S_ajustado = ajustar_tamano_muestra_mejorado(data, S_inicial=40, e_sigma=0.01, IncDat=0.05, reps=10, random_state=42)  # PROPUESTO
print(f"[FIN] HIPERPARAMETERS.JSON ACTUALIZADO")

# LEER O CREAR JSON HIPERPARÁMETROS
if os.path.exists(HIP_JSON):
    with open(HIP_JSON, 'r', encoding='utf-8') as f:
        hip_data = json.load(f)  # CARGAR JSON EXISTENTE
else:
    hip_data = {}  # CREAR NUEVO JSON

# ACTUALIZAR INFORMACIÓN DE S
hip_data['S'] = {
    "value": S_ajustado,  # VALOR AJUSTADO
    "description": "Sample size for Isolation Forest",  # DESCRIPCIÓN
    "adjustment_method": "Standard deviation based incremental sampling",  # MÉTODO AJUSTE
    "default": 256  # VALOR POR DEFECTO
}

# GUARDAR JSON ACTUALIZADO
with open(HIP_JSON, 'w', encoding='utf-8') as f:
    json.dump(hip_data, f, indent=4)  # GUARDAR HIPERPARÁMETROS
