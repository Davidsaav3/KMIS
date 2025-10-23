import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score

def contaminar_dat(Dat, S, porcentaje=0.01, incremento=0.5, random_state=None):
    # FIJA SEMILLA PARA REPRODUCIBILIDAD
    np.random.seed(random_state)
    
    # TOMA UNA MUESTRA DE TAMAÑO S DEL DATASET
    muestra = np.random.choice(Dat, size=S, replace=False)
    
    # DETERMINA EL NÚMERO DE ANOMALÍAS (1% DE LA MUESTRA)
    n_anomalias = max(1, int(S * porcentaje))
    
    # SELECCIONA ALEATORIAMENTE LOS ÍNDICES A CONTAMINAR
    indices_anom = np.random.choice(S, size=n_anomalias, replace=False)
    
    # SELECCIONA UNA CARACTERÍSTICA ALEATORIA (SI ES 1D, ES DIRECTA)
    # SI Dat ES 1D, solo aplica incremento
    contaminada = muestra.copy()
    contaminada[indices_anom] = contaminada[indices_anom] * (1 + incremento)
    
    # RETORNA DATOS CONTAMINADOS Y LOS ÍNDICES DE ANOMALÍAS
    return contaminada, indices_anom

def ajustar_numero_arboles(Dat, S, T_min=5, T_max=100, step=5, N=3, F1sta=0.01, random_state=None):
    # FIJA SEMILLA PARA REPRODUCIBILIDAD
    np.random.seed(random_state)

    # CONTAMINA EL DATASET CON 1% DE ANOMALÍAS
    Dat_contaminado, indices_anom_real = contaminar_dat(Dat, S, porcentaje=0.01, incremento=0.5, random_state=random_state)
    print(f"[INFO] Dataset contaminado con {len(indices_anom_real)} anomalías artificiales")

    T = T_min  # INICIALIZA T
    F1_list = []  # LISTA PARA ALMACENAR F1-SCORES

    while T <= T_max:
        print(f"[INFO] Probando T={T} árboles")

        # EJECUTA ISOLATION FOREST
        IF = IsolationForest(n_estimators=T, contamination=0.01, random_state=random_state)
        IF.fit(Dat_contaminado.reshape(-1, 1))  # ASUME DAT 1D
        scores = -IF.decision_function(Dat_contaminado.reshape(-1, 1))  # PUNTUACIONES DE ANOMALÍA

        # SELECCIONA EL 1% CON MAYOR SCORE COMO ANOMALÍAS PREDICHAS
        n_anom_pred = max(1, int(S * 0.01))
        indices_pred = np.argsort(scores)[-n_anom_pred:]

        # CALCULA F1-SCORE
        y_true = np.zeros(S)
        y_true[indices_anom_real] = 1
        y_pred = np.zeros(S)
        y_pred[indices_pred] = 1
        F1 = f1_score(y_true, y_pred)
        F1_list.append(F1)
        print(f"[INFO] F1-score: {F1:.4f}")

        # COMPRUEBA ESTABILIDAD SI HAY SUFICIENTES ITERACIONES
        if len(F1_list) >= N:
            cumple = sum(1 for f in F1_list[-N:] if f <= F1sta)
            if cumple == N:
                print(f"[INFO] F1-score estable detectado. T final: {T}")
                return T

        # INCREMENTA T
        T += step

    # SI NO SE ESTABILIZA, RETORNA T_MAX
    print(f"[INFO] F1-score no estabilizó. T final = {T_max}")
    return T_max