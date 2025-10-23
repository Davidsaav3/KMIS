import numpy as np

def desviacion_tipica(data):
    # CALCULA LA DESVIACIÓN TÍPICA DEL CONJUNTO DE DATOS
    return np.std(data)

def ajustar_tamano_muestra(Dat, S_inicial=256, eσ=0.05, IncDat=0.1, random_state=None):
    # FIJA SEMILLA PARA REPRODUCIBILIDAD
    np.random.seed(random_state)

    # INICIALIZA EL TAMAÑO DE MUESTRA
    S = S_inicial
    print(f"[INFO] Tamaño inicial de muestra: {S}")

    # CALCULA LA DESVIACIÓN TÍPICA DEL CONJUNTO ORIGINAL
    σo = desviacion_tipica(Dat)
    print(f"[INFO] Desviación típica del conjunto original σo: {σo:.4f}")

    # CALCULA σmin PARA LA MUESTRA INICIAL
    σmin = desviacion_tipica(np.random.choice(Dat, size=min(S, len(Dat)), replace=False))
    print(f"[INFO] Desviación típica de la muestra inicial σmin: {σmin:.4f}")

    # ITERA HASTA QUE σmin ESTÉ DENTRO DE σo ± eσ O SE ALCANCE EL TAMAÑO MÁXIMO
    while S < len(Dat) and not (σo - eσ <= σmin <= σo + eσ):
        # INCREMENTA S
        S = int(min(S * (1 + IncDat), len(Dat)))
        print(f"[INFO] Incrementando tamaño de muestra a: {S}")

        # TOMA MUESTRA ALEATORIA DEL NUEVO S
        muestra = np.random.choice(Dat, size=S, replace=False)

        # CALCULA DESVIACIÓN TÍPICA DE LA NUEVA MUESTRA
        σmin = desviacion_tipica(muestra)
        print(f"[INFO] Desviación típica de la nueva muestra σmin: {σmin:.4f}")

    # RETORNA EL TAMAÑO AJUSTADO
    print(f"[INFO] Tamaño de muestra final ajustado: {S}")
    return S