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
    np.random.seed(random_state)  # FIJAR SEMILLA PARA REPRODUCIBILIDAD
    S = S_inicial  # INICIALIZAR TAMAÑO DE MUESTRA
    print(f"[INFO] Tamaño inicial de muestra: {S}")

    sigma_obj = np.std(Dat)  # DESVIACIÓN ESTANDAR DEL CONJUNTO COMPLETO
    # [ CAMBIO ] En la versión anterior era sigma_o, calculado igual, pero no se usaba promedio de submuestras.

    # [ CAMBIO ] Nueva función interna para calcular desviación promedio de varias submuestras
    # Esto mejora la estabilidad frente a la variabilidad aleatoria.
    def sigma_promedio(S):
        muestras = [np.random.choice(Dat, size=S, replace=False) for _ in range(reps)]  # [ CAMBIO ] varias submuestras
        return np.mean([np.std(m) for m in muestras])  # [ CAMBIO ] promedio de desviaciones en vez de una sola muestra

    sigma_muestra = sigma_promedio(S)  # [ CAMBIO ] Cálculo más robusto de la desviación inicial
    print(f"[INFO] Desviación promedio de la muestra inicial: {sigma_muestra:.4f}")

    # [ CAMBIO ] Bucle optimizado: incremento dinámico según diferencia entre sigma_obj y sigma_muestra
    while S < len(Dat) and not (sigma_obj - e_sigma <= sigma_muestra <= sigma_obj + e_sigma):
        incremento = max(1, int(S * IncDat * abs(sigma_obj - sigma_muestra) / sigma_obj))  
        # [ CAMBIO ] incremento variable, ajustado proporcionalmente al error relativo de sigma
        
        S = min(S + incremento, len(Dat))  # ACTUALIZAR TAMAÑO DE MUESTRA SIN SUPERAR TOTAL
        sigma_muestra = sigma_promedio(S)  # [ CAMBIO ] recalcular con promedio en lugar de una sola muestra

        print(f"[INFO] Incrementando tamaño de muestra a: {S}")  # INFORMAR NUEVO TAMAÑO
        print(f"[INFO] Desviación promedio de la nueva muestra: {sigma_muestra:.4f}")  # INFORMAR NUEVA DESVIACIÓN

    print(f"[INFO] Tamaño de muestra final ajustado: {S}")  # MOSTRAR RESULTADO FINAL
    return S  # RETORNAR TAMAÑO DE MUESTRA AJUSTADO


# CALL
# data → Dataset base sobre el que se calcula el tamaño óptimo de muestra.
# S_inicial → Tamaño de muestra inicial que actúa como punto de partida en el proceso de ajuste.
# e_sigma → Error o tolerancia de convergencia; indica cuándo detener el ajuste (menor valor = mayor precisión).
# IncDat → Porcentaje de contaminación o perturbación introducida en cada iteración para medir estabilidad.
# reps → Número de repeticiones por iteración para obtener resultados más estables y menos dependientes del azar.
# random_state → Semilla aleatoria que garantiza la reproducibilidad del ajuste.

# S_ajustado = ajustar_tamano_muestra(data, S_inicial=256, e_sigma=0.05, IncDat=0.1, random_state=42) # Original
# S_ajustado = ajustar_tamano_muestra(data, S_inicial=40, e_sigma=0.01, IncDat=0.05, random_state=42) # Ajustado
S_ajustado = ajustar_tamano_muestra_mejorado(data, S_inicial=40, e_sigma=0.01, IncDat=0.05, reps=10, random_state=42) # Propuesto
print(f"[FIN] hiperparameters.json actualizado con S={S_ajustado}")


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
