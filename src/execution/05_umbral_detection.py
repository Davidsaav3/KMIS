import numpy as np

def contaminar_dat_5pct(Dat, porcentaje=0.05, incremento=0.5, random_state=None):
    # FIJA SEMILLA PARA REPRODUCIBILIDAD
    np.random.seed(random_state)
    
    S = len(Dat)
    n_anom = max(1, int(S * porcentaje))
    
    # COPIA DEL DATASET
    Dat_contaminado = Dat.copy()
    
    # INDICES A CONTAMINAR
    indices_anom = np.random.choice(S, size=n_anom, replace=False)
    
    # INCREMENTA 50% EN LOS REGISTROS SELECCIONADOS
    Dat_contaminado[indices_anom] = Dat_contaminado[indices_anom] * (1 + incremento)
    
    return Dat_contaminado, indices_anom

def calcular_FC(Dat_cont, indices_anom, Th, delta=0.2):
    # CALCULA FALSOS POSITIVOS Y FALSOS NEGATIVOS SEGUN UMBRAL Th
    scores = Dat_cont  # ASUMIMOS QUE scores = valores del dataset para simplicidad
    pred = np.zeros(len(Dat_cont))
    pred[scores >= Th] = 1  # CLASIFICA COMO ANOMALÍA SI SCORE >= Th

    # VECTOR DE VERDAD
    y_true = np.zeros(len(Dat_cont))
    y_true[indices_anom] = 1

    FP = np.sum((pred == 1) & (y_true == 0)) / max(1, np.sum(y_true == 0))  # TASA DE FALSOS POSITIVOS
    FN = np.sum((pred == 0) & (y_true == 1)) / max(1, np.sum(y_true == 1))  # TASA DE FALSOS NEGATIVOS

    # FUNCION DE COSTE
    FC = delta * FP + (1 - delta) * FN
    return FC

def ajustar_umbral(Dat, delta=0.2, Th_min=0.0, Th_max=1.0, grad=0.01, Th=0.5, random_state=None):
    np.random.seed(random_state)

    # CONTAMINA 5% DEL DATASET
    Dat_cont, indices_anom = contaminar_dat_5pct(Dat, porcentaje=0.05, incremento=0.5, random_state=random_state)
    print(f"[INFO] Dataset contaminado con {len(indices_anom)} anomalías")

    # BÚSQUEDA BINARIA
    while Th_max - Th_min >= grad:
        mid1 = (Th + Th_min) / 2
        mid2 = (Th_max + Th) / 2

        FC1 = calcular_FC(Dat_cont, indices_anom, mid1, delta)
        FC2 = calcular_FC(Dat_cont, indices_anom, mid2, delta)

        print(f"[INFO] Th={Th:.4f}, mid1={mid1:.4f}, FC1={FC1:.4f}, mid2={mid2:.4f}, FC2={FC2:.4f}")

        if FC1 < FC2:
            Th_max = Th
            Th = mid1
        else:
            Th_min = Th
            Th = mid2

    print(f"[INFO] Umbral de detección ajustado: Th={Th:.4f}")
    return Th