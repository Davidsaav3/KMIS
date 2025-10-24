import numpy as np 
import pandas as pd
from sklearn.ensemble import IsolationForest
from math import log2
import json
import os

# PARÁMETROS DE ENTRADA Y SALIDA
INPUT_CSV = '../../results/preparation/05_variance_recortado.csv'  # RUTA DEL CSV DE DATOS
HIP_JSON = '../../results/execution/hiperparameters.json'  # RUTA DEL JSON DE HIPERPARÁMETROS

# CARGAR DATASET
df = pd.read_csv(INPUT_CSV)  # LEER CSV DESDE DISCO

# ELIMINAR COLUMNA 'is_anomaly' SI EXISTE
if 'is_anomaly' in df.columns:
    df = df.drop(columns=['is_anomaly'])  # ELIMINAR COLUMNA NO RELEVANTE

# CONVERTIR DATOS NUMÉRICOS A VECTOR 1D
Dat_np = df.select_dtypes(include=[np.number]).values.flatten()  # EXTRAER Y APLANAR DATOS NUMÉRICOS

# CONTAMINA 5% DE LOS DATOS AUMENTANDO SUS VALORES
def contaminar_dat_5pct(Dat, porcentaje=0.05, incremento=0.5, random_state=None):
    np.random.seed(random_state)
    S = len(Dat)
    n_anom = max(1, int(S * porcentaje))
    Dat_cont = Dat.copy()
    indices_anom = np.random.choice(S, size=n_anom, replace=False)
    Dat_cont[indices_anom] *= (1 + incremento)
    return Dat_cont, indices_anom


# FUNCIÓN PARA AJUSTAR EL FACTOR D (PROFUNDIDAD EFECTIVA)
def ajustar_profundidad_maxima(Dat, S, betha=0.2, random_state=None, max_iter=50):
    # AJUSTA D SEGÚN LA TASA PROMEDIO DE AISLAMIENTO
    np.random.seed(random_state)  # FIJAR SEMILLA ALEATORIA
    # CONTAMINAR SUBMUESTRA CON ANOMALÍAS
    Dat_cont, indices_anom = contaminar_dat_5pct(Dat[:S], porcentaje=0.05, incremento=0.5, random_state=random_state)
    print(f"[INFO] Dataset contaminado con {len(indices_anom)} anomalías")

    # DEFINIR RANGO INICIAL DE PROFUNDIDAD
    D_max = int(log2(S))  # PROFUNDIDAD MÁXIMA TEÓRICA
    D_min = 1  # PROFUNDIDAD MÍNIMA PERMITIDA
    D = D_max  # INICIALIZAR CON D_MAX
    print(f"[INFO] D inicial: {D}")

    # CALCULAR REFERENCIA DE AISLAMIENTO CON D INICIAL
    IF = IsolationForest(max_samples=S, max_features=1.0, n_estimators=100, random_state=random_state)
    IF.fit(Dat_cont.reshape(-1, 1))  # ENTRENAR MODELO CON DATOS CONTAMINADOS
    scores = -IF.decision_function(Dat_cont.reshape(-1, 1))  # MAYOR SCORE = MÁS ANÓMALO
    n_anom = max(1, int(S * 0.05))  # CANTIDAD DE ANOMALÍAS A CONSIDERAR
    top_indices = np.argsort(scores)[-n_anom:]  # ÍNDICES DE MAYOR SCORE
    CR = scores[top_indices]  # SCORES DE ANOMALÍAS DETECTADAS
    R_25, R_75 = np.percentile(CR, [25, 75])  # CALCULAR CUARTILES
    print(f"[INFO] R_25={R_25:.4f}, R_75={R_75:.4f}")

    # INICIAR BUCLE DE AJUSTE ITERATIVO
    iteration = 0
    while iteration < max_iter:
        iteration += 1
        print(f"[INFO] Iteración {iteration}: Probando factor D={D}")
        max_samples = max(1, int(S * D / D_max))  # AJUSTAR TAMAÑO DE MUESTRA SEGÚN D
        IF = IsolationForest(max_samples=max_samples, max_features=1.0, n_estimators=100, random_state=random_state)
        IF.fit(Dat_cont.reshape(-1, 1))  # REENTRENAR MODELO
        scores = -IF.decision_function(Dat_cont.reshape(-1, 1))  # NUEVOS SCORES
        top_indices = np.argsort(scores)[-n_anom:]  # ANOMALÍAS MÁS DESTACADAS
        CR = scores[top_indices]  # EXTRAER SCORES DE INTERÉS
        R = np.mean(CR)  # CALCULAR TASA PROMEDIO DE AISLAMIENTO
        print(f"[INFO] Tasa promedio de aislamiento R={R:.4f}")

        # AJUSTAR D SEGÚN POSICIÓN DE R EN CUARTILES
        new_D = D
        if R < R_75 and D > D_min:
            new_D = max(int(D * (1 - betha)), D_min)  # REDUCIR PROFUNDIDAD
        elif R > R_25 and D < D_max:
            new_D = min(int(D * (1 + betha)), D_max)  # AUMENTAR PROFUNDIDAD

        # DETENER SI D SE ESTABILIZA
        if new_D == D:
            print(f"[INFO] Factor D ajustado encontrado: {D}")
            return D

        D = new_D  # ACTUALIZAR D

    # FINALIZAR SI SE ALCANZA EL LÍMITE DE ITERACIONES
    print(f"[WARN] Máximo número de iteraciones alcanzado, factor D final: {D}")
    return D


def ajustar_profundidad_maxima_optimo(Dat, S, betha=0.2, random_state=None, max_iter=50, reps=5):
    """Ajusta la profundidad máxima D del Isolation Forest de manera más robusta"""

    np.random.seed(random_state)
    Dat_cont, indices_anom = contaminar_dat_5pct(Dat[:S], porcentaje=0.05, incremento=0.5, random_state=random_state)
    print(f"[INFO] Dataset contaminado con {len(indices_anom)} anomalías")

    D_max = int(log2(S))
    D_min = 1
    D = D_max
    print(f"[INFO] D inicial: {D}")

    # CALCULAR CUARTILES BASE USANDO MÁS ÁRBOLES → mayor estabilidad estadística
    IF = IsolationForest(max_samples=S, n_estimators=150, random_state=random_state)  # ↑ CAMBIO: 150 árboles
    IF.fit(Dat_cont.reshape(-1, 1))
    scores = -IF.decision_function(Dat_cont.reshape(-1, 1))
    n_anom = max(1, int(S * 0.05))
    top_indices = np.argsort(scores)[-n_anom:]
    CR = scores[top_indices]
    R_25, R_75 = np.percentile(CR, [25, 75])
    print(f"[INFO] R_25={R_25:.4f}, R_75={R_75:.4f}")

    last_R = None
    stable_count = 0

    for iteration in range(1, max_iter + 1):
        print(f"[INFO] Iteración {iteration}: Probando factor D={D}")
        max_samples = max(1, int(S * D / D_max))

        # PROMEDIO SOBRE VARIAS REPETICIONES → reduce varianza en R
        R_list = []
        for r in range(reps):  # ↑ CAMBIO: se repite reps veces
            IF = IsolationForest(max_samples=max_samples, n_estimators=150, random_state=random_state + r)
            IF.fit(Dat_cont.reshape(-1, 1))
            scores = -IF.decision_function(Dat_cont.reshape(-1, 1))
            top_indices = np.argsort(scores)[-n_anom:]
            R_list.append(np.mean(scores[top_indices]))

        R = np.mean(R_list)  # PROMEDIO DE LAS TASAS R
        print(f"[INFO] Tasa promedio de aislamiento R={R:.4f}")

        # CRITERIO DE CONVERGENCIA → si R se estabiliza, detenemos
        if last_R is not None and abs(R - last_R) < 0.005:  # ↑ CAMBIO: tolerancia de convergencia
            stable_count += 1
            if stable_count >= 3:  # ↑ CAMBIO: 3 iteraciones consecutivas estables
                print(f"[INFO] Convergencia alcanzada. Factor D final: {D}")
                return D
        else:
            stable_count = 0
        last_R = R

        # AJUSTE ADAPTATIVO DE BETHA → se reduce si hay oscilaciones
        if iteration > 1 and stable_count == 0:
            betha = max(0.1, betha * 0.9)  # ↑ CAMBIO: reducción progresiva del paso

        # NUEVAS REGLAS DE AJUSTE → más flexibles y suaves
        new_D = D
        if R < R_75 and D > D_min:
            new_D = max(int(D * (1 - betha)), D_min)  # REDUCE D si R bajo
        elif R > R_25 * 0.95 and D < D_max:  # ↑ CAMBIO: permite leve aumento si mejora
            new_D = min(int(D * (1 + betha)), D_max)

        # CRITERIO FINAL DE PARADA
        if new_D == D:
            print(f"[INFO] Factor D ajustado encontrado: {D}")
            return D

        D = new_D  # ACTUALIZA D PARA LA SIGUIENTE ITERACIÓN

    print(f"[WARN] Máximo número de iteraciones alcanzado, factor D final: {D}")
    return D


# LEER S DESDE JSON
if os.path.exists(HIP_JSON):
    with open(HIP_JSON, 'r', encoding='utf-8') as f:
        hip_data = json.load(f)
    S = hip_data.get('S', {}).get('value', 200)  # Si no existe, usar 200 como default
else:
    S = 256  # Default si no existe JSON


# MAIN
# D_ajustado = ajustar_profundidad_maxima(Dat_np, S, betha=0.2, random_state=42)
D_ajustado = ajustar_profundidad_maxima_optimo(Dat_np, S, betha=0.2, random_state=42)
print(f"[FIN] hiperparameters.json actualizado con D={D_ajustado}")


# CARGAR O CREAR JSON DE HIPERPARÁMETROS
if os.path.exists(HIP_JSON):
    with open(HIP_JSON, 'r', encoding='utf-8') as f:
        hip_data = json.load(f)  # LEER JSON EXISTENTE
else:
    hip_data = {}  # CREAR NUEVO SI NO EXISTE
# ACTUALIZAR O AÑADIR VALOR DE D
hip_data['D'] = {
    "value": D_ajustado,  # VALOR FINAL DE D
    "description": "Factor controlling Isolation Forest effective depth",  # DESCRIPCIÓN
    "adjustment_method": "Adjusted using average isolation rate R and quartiles",  # MÉTODO DE AJUSTE
    "default": "log2(S)"  # VALOR POR DEFECTO
}
# GUARDAR JSON ACTUALIZADO
with open(HIP_JSON, 'w', encoding='utf-8') as f:
    json.dump(hip_data, f, indent=4)  # ESCRIBIR JSON FORMATEADO