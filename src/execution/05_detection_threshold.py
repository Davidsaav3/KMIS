import numpy as np 
import pandas as pd
import json
import os

# PARÁMETROS DE ENTRADA Y SALIDA
INPUT_CSV = '../../results/preparation/05_variance_recortado.csv'  # CSV DE DATOS NUMÉRICOS
HIP_JSON = '../../results/execution/hiperparameters.json'  # JSON DE HIPERPARÁMETROS

# CARGAR DATASET
df = pd.read_csv(INPUT_CSV)  # LEER CSV DESDE DISCO

# ELIMINAR COLUMNA 'is_anomaly' SI EXISTE
if 'is_anomaly' in df.columns:
    df = df.drop(columns=['is_anomaly'])  # QUITAR COLUMNA NO NECESARIA PARA AJUSTE

# CONVERTIR COLUMNAS NUMÉRICAS A VECTORIZADO 1D
Dat_np = df.select_dtypes(include=[np.number]).values.flatten()  # EXTRAER Y APLANAR DATOS NUMÉRICOS


# [ CONTAMINAR 5% ] 
def contaminar_dat_5pct(Dat, porcentaje=0.05, incremento=0.5, random_state=None):
    np.random.seed(random_state)  # FIJAR SEMILLA PARA REPRODUCIBILIDAD
    S = len(Dat)  # TAMAÑO TOTAL DE DATOS
    n_anom = max(1, int(S * porcentaje))  # NÚMERO DE ANOMALÍAS A CREAR
    Dat_contaminado = Dat.copy()  # COPIA DE DATOS PARA NO MODIFICAR ORIGINAL
    indices_anom = np.random.choice(S, size=n_anom, replace=False)  # ÍNDICES ALEATORIOS PARA ANOMALÍAS
    Dat_contaminado[indices_anom] *= (1 + incremento)  # INCREMENTAR VALORES SELECCIONADOS
    return Dat_contaminado, indices_anom  # DEVOLVER DATOS CONTAMINADOS Y SUS ÍNDICES


# [ FUNCIÓN DE COSTE ] 
def calcular_FC(Dat_cont, indices_anom, Th, delta=0.2):
    pred = np.zeros(len(Dat_cont))  # INICIALIZAR PREDICCIONES
    pred[Dat_cont >= Th] = 1  # MARCAR ANOMALÍAS SEGÚN UMBRAL
    y_true = np.zeros(len(Dat_cont))  # INICIALIZAR VERDADEROS
    y_true[indices_anom] = 1  # MARCAR ANOMALÍAS REALES
    FP = np.sum((pred == 1) & (y_true == 0)) / max(1, np.sum(y_true == 0))  # CALCULAR FALSOS POSITIVOS
    FN = np.sum((pred == 0) & (y_true == 1)) / max(1, np.sum(y_true == 1))  # CALCULAR FALSOS NEGATIVOS
    FC = delta * FP + (1 - delta) * FN  # PONDERAR FP Y FN
    return FC  # DEVOLVER COSTE


# [ MAIN ] 
def ajustar_umbral(Dat, delta=0.2, Th_min=0.0, Th_max=1.0, grad=0.01, Th=0.5, random_state=None):
    np.random.seed(random_state)  # FIJAR SEMILLA
    Dat_cont, indices_anom = contaminar_dat_5pct(Dat, porcentaje=0.05, incremento=0.5, random_state=random_state)  # CONTAMINAR DATOS
    print(f"[INFO] Dataset contaminado con {len(indices_anom)} anomalías")

    # BÚSQUEDA BINARIA DEL UMBRAL
    while Th_max - Th_min >= grad:  # MIENTRAS EL RANGO SEA MAYOR QUE EL GRADIENTE
        mid1 = (Th + Th_min) / 2  # PUNTO INTERMEDIO INFERIOR
        mid2 = (Th_max + Th) / 2  # PUNTO INTERMEDIO SUPERIOR

        FC1 = calcular_FC(Dat_cont, indices_anom, mid1, delta)  # CALCULAR COSTE EN MID1
        FC2 = calcular_FC(Dat_cont, indices_anom, mid2, delta)  # CALCULAR COSTE EN MID2

        print(f"[INFO] Th={Th:.4f}, mid1={mid1:.4f}, FC1={FC1:.4f}, mid2={mid2:.4f}, FC2={FC2:.4f}")

        if FC1 < FC2:  # ELEGIR LADO CON COSTE MENOR
            Th_max = Th
            Th = mid1
        else:
            Th_min = Th
            Th = mid2

    print(f"[INFO] Umbral de detección ajustado: Th={Th:.4f}")
    return Th  # DEVOLVER UMBRAL AJUSTADO


# [ MAIN MEJORADO ] 
def ajustar_umbral_mejorado(Dat, grad=0.001, Th_min=0.0, Th_max=1.0, Th=None, random_state=None):
    np.random.seed(random_state)  # [ CAMBIO ] FIJAR SEMILLA PARA REPRODUCIBILIDAD

    while Th_max - Th_min >= grad:
        mid = (Th_min + Th_max) / 2  # [ CAMBIO ] SIMPLIFICAR A UN MID
        pct = np.mean(Dat > mid)     # [ CAMBIO ] PROPORCIÓN DE DATOS > UMBRAL

        # [ CAMBIO ] AJUSTE PARA QUE SOLO ~1% DE DATOS QUEDEN POR ENCIMA
        if pct > 0.01:
            Th_min = mid
        else:
            Th_max = mid

        Th = mid  # [ CAMBIO ] ACTUALIZAR TH DIRECTAMENTE

    porcentaje_estimado = np.mean(Dat > Th)
    print(f"[INFO] PORCENTAJE ESTIMADO: {porcentaje_estimado:.6f}, UMBRAL: {Th:.6f}")  # [ CAMBIO ] IMPRIMIR PORCENTAJE Y UMBRAL
    return porcentaje_estimado, Th  # [ CAMBIO ] DEVOLVER TAMBIÉN EL PORCENTAJE


# [ HIPERPARÁMETROS ]
# Dat_np → DATASET DE ENTRADA EN FORMATO NUMPY
# delta → FACTOR DE TOLERANCIA PARA CALCULO DE COSTE
# grad → PRECISIÓN MÍNIMA PARA DETENER BÚSQUEDA BINARIA
# Th_min → VALOR MÍNIMO DEL UMBRAL
# Th_max → VALOR MÁXIMO DEL UMBRAL
# Th → VALOR INICIAL DEL UMBRAL
# random_state → SEMILLA ALEATORIA

# Th_ajustado = ajustar_umbral(Dat_np, delta=0.2, Th_min=0.0, Th_max=1.0, grad=0.01, Th=0.5, random_state=42) # Original
# Th_ajustado = ajustar_umbral(Dat_np, delta=0.2, Th_min=0.0, Th_max=1.0, grad=0.01, Th=0.5, random_state=42) # Ajustado
Th_ajustado, Th_referencia = ajustar_umbral_mejorado(Dat_np, grad=0.005, Th_min=0.0, Th_max=1.0, Th=None, random_state=42)
print(f"[FIN] HIPERPARAMETERS.JSON ACTUALIZADO CON Th={Th_ajustado}")

# CREAR O ACTUALIZAR JSON DE HIPERPARÁMETROS
if os.path.exists(HIP_JSON):
    with open(HIP_JSON, 'r', encoding='utf-8') as f:
        hip_data = json.load(f)  # CARGAR JSON EXISTENTE
else:
    hip_data = {}  # CREAR NUEVO JSON

# GUARDAR VALOR AJUSTADO DE Th
hip_data['Th'] = {
    "value": Th_ajustado,
    "description": "DETECTION THRESHOLD TO CLASSIFY ANOMALIES",
    "adjustment_method": "BINARY SEARCH MINIMIZING COST FUNCTION (FP,FN)",
    "default": 1.0
}

# GUARDAR JSON FORMATEADO
with open(HIP_JSON, 'w', encoding='utf-8') as f:
    json.dump(hip_data, f, indent=4)  # GUARDAR JSON ACTUALIZADO
