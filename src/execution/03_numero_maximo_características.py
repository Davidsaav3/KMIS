import numpy as np
from sklearn.preprocessing import MinMaxScaler

def ajustar_numero_caracteristicas(Dat, F_inicial=1.0, α_reduccion=0.5, α_aumento=1.5, random_state=None):
    # FIJA SEMILLA PARA REPRODUCIBILIDAD
    np.random.seed(random_state)

    # NORMALIZA DAT A [0,1]
    scaler = MinMaxScaler()
    Dat_norm = scaler.fit_transform(Dat)
    print("[INFO] Dataset normalizado a rango [0,1]")

    # NUMERO DE CARACTERÍSTICAS
    num_features = Dat.shape[1]
    F_min = 1 / num_features
    F_max = 1.0

    # INICIALIZA F
    F = F_inicial
    print(f"[INFO] F inicial: {F:.4f}")

    while True:
        # SELECCIONA ALEATORIAMENTE F * num_features CARACTERÍSTICAS
        n_selected = max(1, int(F * num_features))
        selected_idx = np.random.choice(num_features, size=n_selected, replace=False)
        selected_features = Dat_norm[:, selected_idx]

        # CALCULA VARIANZA DE CADA CARACTERÍSTICA SELECCIONADA
        σ2 = np.var(selected_features, axis=0)

        # PROMEDIO DE LAS VARIANZAS
        V_prom = np.mean(σ2)

        # CALCULA CUARTILES
        Q1, Q2, Q3, Q4 = np.percentile(σ2, [25, 50, 75, 100])

        # CALCULA FRONTERAS BQ1F Y BQ4F
        BQ1F = (np.max([Q1]) + np.min([Q2])) / 2
        BQ4F = (np.max([Q3]) + np.min([Q4])) / 2

        print(f"[INFO] V_prom={V_prom:.4f}, BQ1F={BQ1F:.4f}, BQ4F={BQ4F:.4f}, F={F:.4f}")

        # AJUSTA F SEGÚN V_prom
        if V_prom > BQ4F:
            # ALTA DISPERSIÓN, REDUCIR F
            if F * α_reduccion < F_min:
                print("[INFO] F alcanzó F_min, se retorna F")
                return F
            F *= α_reduccion
            print(f"[INFO] Reducción de F a {F:.4f}")
        elif V_prom < BQ1F:
            # BAJA DISPERSIÓN, AUMENTAR F
            if F * α_aumento > F_max:
                print("[INFO] F alcanzó F_max, se retorna F")
                return F
            F *= α_aumento
            print(f"[INFO] Aumento de F a {F:.4f}")
        else:
            # F ESTABLE
            print("[INFO] F ajustado encontrado")
            return F

        # AQUÍ SE PODRÍA ACTUALIZAR MODELO IF CON NUEVO F
        # actualizar_modelo_IF(F)  # OPCIONAL