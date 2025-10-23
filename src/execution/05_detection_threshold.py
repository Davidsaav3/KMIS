import numpy as np 
import pandas as pd
import json
import os

# PARÁMETROS DE ENTRADA Y SALIDA
INPUT_CSV = '../../results/execution/00_contaminated.csv'  # CSV DE DATOS
HIP_JSON = '../../results/execution/hiperparameters.json'  # JSON DE HIPERPARÁMETROS

# CARGAR DATASET
df = pd.read_csv(INPUT_CSV)  # LEER CSV

# ELIMINAR COLUMNA 'is_anomaly' SI EXISTE
if 'is_anomaly' in df.columns:
    df = df.drop(columns=['is_anomaly'])  # QUITAR COLUMNA NO NECESARIA

# CONVERTIR COLUMNAS NUMÉRICAS A VECTORIZADO 1D
Dat_np = df.select_dtypes(include=[np.number]).values.flatten()  # EXTRAER DATOS NUMÉRICOS

# FUNCIONES

def contaminar_dat_5pct(Dat, porcentaje=0.05, incremento=0.5, random_state=None):
    """CONTAMINA 5% DEL DATASET CON ANOMALÍAS ARTIFICIALES"""
    np.random.seed(random_state)  # FIJAR SEMILLA ALEATORIA
    S = len(Dat)  # TAMAÑO TOTAL DE DATOS
    n_anom = max(1, int(S * porcentaje))  # CALCULAR NÚMERO DE ANOMALÍAS
    Dat_contaminado = Dat.copy()  # COPIA DE DATOS PARA CONTAMINAR
    indices_anom = np.random.choice(S, size=n_anom, replace=False)  # SELECCIONAR ÍNDICES ALEATORIOS
    Dat_contaminado[indices_anom] *= (1 + incremento)  # INCREMENTAR VALORES SELECCIONADOS
    return Dat_contaminado, indices_anom  # DEVOLVER DATOS CONTAMINADOS Y ÍNDICES

def calcular_FC(Dat_cont, indices_anom, Th, delta=0.2):
    """CALCULA FUNCION DE COSTE FC (FP Y FN PONDERADOS)"""
    pred = np.zeros(len(Dat_cont))  # INICIALIZAR PREDICCIONES
    pred[Dat_cont >= Th] = 1  # MARCAR ANOMALÍAS SEGÚN UMBRAL
    y_true = np.zeros(len(Dat_cont))  # INICIALIZAR VERDADEROS
    y_true[indices_anom] = 1  # MARCAR ANOMALÍAS REALES
    FP = np.sum((pred == 1) & (y_true == 0)) / max(1, np.sum(y_true == 0))  # CALCULAR FALSOS POSITIVOS
    FN = np.sum((pred == 0) & (y_true == 1)) / max(1, np.sum(y_true == 1))  # CALCULAR FALSOS NEGATIVOS
    FC = delta * FP + (1 - delta) * FN  # PONDERAR FP Y FN
    return FC  # DEVOLVER FC

def ajustar_umbral(Dat, delta=0.2, Th_min=0.0, Th_max=1.0, grad=0.01, Th=0.5, random_state=None):
    """AJUSTA UMBRAL DE DETECCIÓN USANDO BÚSQUEDA BINARIA SOBRE FC"""
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

# EJECUCIÓN DEL AJUSTE
Th_ajustado = ajustar_umbral(Dat_np, delta=0.2, Th_min=0.0, Th_max=1.0, grad=0.01, Th=0.5, random_state=42)
print(f"[INFO] Umbral final ajustado: Th={Th_ajustado}")

# CREAR O ACTUALIZAR JSON DE HIPERPARÁMETROS
if os.path.exists(HIP_JSON):
    with open(HIP_JSON, 'r', encoding='utf-8') as f:
        hip_data = json.load(f)  # CARGAR EXISTENTE
else:
    hip_data = {}  # CREAR NUEVO

# GUARDAR VALOR AJUSTADO DE Th
hip_data['Th'] = {
    "value": Th_ajustado,
    "description": "Detection threshold to classify anomalies",
    "adjustment_method": "Binary search minimizing cost function (FP,FN)",
    "default": 1.0
}

with open(HIP_JSON, 'w', encoding='utf-8') as f:
    json.dump(hip_data, f, indent=4)  # GUARDAR JSON

print(f"[INFO] hiperparameters.json actualizado con Th={Th_ajustado}")
