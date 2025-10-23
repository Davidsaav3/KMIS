import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from math import log2

INPUT_CSV = '../../results/execution/00_contaminated.csv'
df = pd.read_csv(INPUT_CSV)

def contaminar_dat_5pct(Dat, porcentaje=0.05, incremento=0.5, random_state=None):
    # FIJA SEMILLA PARA REPRODUCIBILIDAD
    np.random.seed(random_state)
    
    S = len(Dat)
    n_anom = max(1, int(S * porcentaje))
    
    # COPIA DEL DATASET
    Dat_contaminado = Dat.copy()
    
    # INDICES A CONTAMINAR
    indices_anom = np.random.choice(S, size=n_anom, replace=False)
    
    # SELECCIONA UNA CARACTERÍSTICA ALEATORIA (1D ASUME DAT UNIDIMENSIONAL)
    Dat_contaminado[indices_anom] = Dat_contaminado[indices_anom] * (1 + incremento)
    
    return Dat_contaminado, indices_anom

def ajustar_profundidad_maxima(Dat, S, β=0.2, random_state=None):
    np.random.seed(random_state)

    # CONTAMINA 5% DE LA MUESTRA
    Dat_cont, indices_anom = contaminar_dat_5pct(Dat[:S], porcentaje=0.05, incremento=0.5, random_state=random_state)
    print(f"[INFO] Dataset contaminado con {len(indices_anom)} anomalías")

    D_max = int(log2(S))
    D_min = 1
    D = D_max
    print(f"[INFO] D inicial: {D}")

    # CALCULA IF INICIAL PARA OBTENER R_25 Y R_75
    IF = IsolationForest(max_samples=S, max_features=1.0, n_estimators=100, random_state=random_state)
    IF.fit(Dat_cont.reshape(-1, 1))
    scores = -IF.decision_function(Dat_cont.reshape(-1, 1))
    # CONSIDERAMOS QUE LAS ANOMALÍAS ESTÁN EN LOS 5% CON MAYOR SCORE
    n_anom = max(1, int(S * 0.05))
    top_indices = np.argsort(scores)[-n_anom:]
    CR = scores[top_indices]
    R_25, R_75 = np.percentile(CR, [25, 75])
    print(f"[INFO] R_25={R_25:.4f}, R_75={R_75:.4f}")

    # ITERA AJUSTANDO D
    while True:
        print(f"[INFO] Probando D={D}")
        IF = IsolationForest(max_depth=D, max_samples=S, max_features=1.0, n_estimators=100, random_state=random_state)
        IF.fit(Dat_cont.reshape(-1, 1))
        scores = -IF.decision_function(Dat_cont.reshape(-1, 1))
        top_indices = np.argsort(scores)[-n_anom:]
        CR = scores[top_indices]
        R = np.mean(CR)
        print(f"[INFO] Tasa promedio de aislamiento R={R:.4f}")

        # AJUSTE SEGÚN R Y CUARTILES
        if R < R_75 and D > D_min:
            # REDUCIR D
            D = max(int(D * (1 - β)), D_min)
            print(f"[INFO] Reducción de D a {D}")
        elif R > R_25 and D < D_max:
            # AUMENTAR D
            D = min(int(D * (1 + β)), D_max)
            print(f"[INFO] Aumento de D a {D}")
        else:
            print(f"[INFO] D ajustado encontrado: {D}")
            return D

# --- EJECUCIÓN ---
S = 256  # TAMAÑO DE MUESTRA PREVIAMENTE CALCULADO
D = D_ajustado = ajustar_profundidad_maxima(df, S, β=0.2, random_state=42)