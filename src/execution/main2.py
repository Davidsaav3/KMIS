import numpy as np
from math import log2

# --- IMPORTAR FUNCIONES PREVIAS ---
# Suponemos que las funciones están definidas o importadas:
# ajustar_tamano_muestra, ajustar_numero_arboles, ajustar_numero_caracteristicas,
# ajustar_profundidad_maxima, ajustar_umbral

# --- DATASETS DE EJEMPLO ---
np.random.seed(42)
Dat_1D = np.random.randn(1000)        # DATASET 1D PARA S, T, D, Th
Dat_multi = np.random.rand(1000, 10)  # DATASET MULTIVARIANTE PARA F

# --- MAIN ---
def main():
    print("[INFO] --- INICIO DEL AJUSTE AUTOMÁTICO DE HIPERPARÁMETROS ---")

    # --- 1. AJUSTE DEL TAMAÑO DE MUESTRA S ---
    # SE USA VALOR POR DEFECTO PARA LOS DEMÁS HP (T=100, F=1.0, D=log2(S), Th=1.0)
    S = ajustar_tamano_muestra(Dat_1D, S_inicial=256, eσ=0.05, IncDat=0.1, random_state=42)

    # --- 2. AJUSTE DEL NÚMERO DE ÁRBOLES T ---
    # SE USA S ÓPTIMO Y DEMÁS VALORES POR DEFECTO
    T = ajustar_numero_arboles(Dat_1D, S, T_min=5, T_max=100, step=5, N=3, F1sta=0.01, random_state=42)

    # --- 3. AJUSTE DEL NÚMERO MÁXIMO DE CARACTERÍSTICAS F ---
    # SE USA S Y T ÓPTIMOS, DEMÁS HP POR DEFECTO
    F = ajustar_numero_caracteristicas(Dat_multi[:S, :], F_inicial=1.0, α_reduccion=0.5, α_aumento=1.5, random_state=42)

    # --- 4. AJUSTE DE LA PROFUNDIDAD MÁXIMA D ---
    # SE USA S, T Y F ÓPTIMOS, Th POR DEFECTO
    D = ajustar_profundidad_maxima(Dat_1D, S, β=0.2, random_state=42)

    # --- 5. AJUSTE DEL UMBRAL DE DETECCIÓN Th ---
    # SE USA TODOS LOS HP ÓPTIMOS PREVIOS
    Th = ajustar_umbral(Dat_1D[:S], delta=0.2, Th_min=0.0, Th_max=1.0, grad=0.01, Th=0.5, random_state=42)

    # --- RESULTADOS FINALES ---
    print("[INFO] --- AJUSTE COMPLETADO ---")
    print(f"[INFO] Parámetros finales ajustados:")
    print(f"       S  (tamaño de muestra)      = {S}")
    print(f"       T  (número de árboles)      = {T}")
    print(f"       F  (número máximo de feats) = {F:.4f}")
    print(f"       D  (profundidad máxima)     = {D}")
    print(f"       Th (umbral de detección)   = {Th:.4f}")

# --- EJECUCIÓN DEL MAIN ---
if __name__ == "__main__":
    main()
