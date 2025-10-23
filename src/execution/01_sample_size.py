import pandas as pd
import numpy as np

INPUT_CSV = '../../results/execution/00_contaminated.csv'
df = pd.read_csv(INPUT_CSV)

# CONVERTIR A ARRAY NUMÉRICO (TODAS LAS COLUMNAS NUMÉRICAS)
data = df.select_dtypes(include=[np.number]).values.flatten()  # APLANAR A 1D

def desviacion_tipica(data):
    # CALCULA LA DESVIACIÓN TÍPICA DE UN VECTOR 1D
    return np.std(data)

def ajustar_tamano_muestra(Dat, S_inicial=256, eσ=0.05, IncDat=0.1, random_state=None):
    np.random.seed(random_state)
    S = S_inicial
    print(f"[INFO] Tamaño inicial de muestra: {S}")

    # σo: desviación típica del conjunto original
    σo = desviacion_tipica(Dat)
    print(f"[INFO] Desviación típica del conjunto original σo: {σo:.4f}")

    # σmin: desviación típica de la muestra inicial
    σmin = desviacion_tipica(np.random.choice(Dat, size=min(S, len(Dat)), replace=False))
    print(f"[INFO] Desviación típica de la muestra inicial σmin: {σmin:.4f}")

    # ITERAR HASTA QUE σmin ESTÉ DENTRO DE σo ± eσ O SE ALCANCE EL TAMAÑO MÁXIMO
    while S < len(Dat) and not (σo - eσ <= σmin <= σo + eσ):
        S = int(min(S * (1 + IncDat), len(Dat)))
        print(f"[INFO] Incrementando tamaño de muestra a: {S}")

        muestra = np.random.choice(Dat, size=S, replace=False)
        σmin = desviacion_tipica(muestra)
        print(f"[INFO] Desviación típica de la nueva muestra σmin: {σmin:.4f}")

    print(f"[INFO] Tamaño de muestra final ajustado: {S}")
    return S

# --- EJECUCIÓN ---
S = ajustar_tamano_muestra(data, S_inicial=256, eσ=0.05, IncDat=0.1, random_state=42)