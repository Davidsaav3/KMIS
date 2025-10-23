import pandas as pd
import numpy as np

# CONFIGURACIÓN DE PARÁMETROS
INPUT_CSV = '../../results/preparation/05_variance_recortado.csv'       # ARCHIVO DE ENTRADA DEL DATASET
OUTPUT_CSV = '../../results/execution/00_contaminated.csv'    # ARCHIVO DE SALIDA DEL DATASET CONTAMINADO
RESULTS_FOLDER = '../../results/preparation'                  # CARPETA PARA RESULTADOS INTERMEDIOS

CONTAMINATION_RATE = 0.01
# PORCENTAJE DE FILAS ANÓMALAS (1%)
# DEFINE CUÁNTOS DATOS SERÁN CONSIDERADOS ANOMALÍAS
# MIN: 0.001, MAX: 0.1, RECOMENDADO: 0.01–0.05
# AFECTA SENSIBILIDAD Y DEFINICIÓN DE VALORES EXTREMOS

NOISE_INTENSITY = 3.0
# INTENSIDAD DEL RUIDO APLICADO A LAS ANOMALÍAS
# CUANTO MAYOR SEA, MAYORES DESVIACIONES RESPECTO A VALORES NORMALES
# MIN: 0.1, MAX: 5–10, RECOMENDADO: 2–4
# AFECTA MÁXIMOS, MÍNIMOS Y DETECTABILIDAD DE LAS ANOMALÍAS

OPERACION_RUIDO = 'SUMA'
# OPERACIÓN PARA GENERAR RUIDO: 'SUMA', 'RESTA', 'MULTIPLICACION', 'ESCALA'
# DEFINE CÓMO SE MODIFICAN LOS VALORES ORIGINALES

COLUMNS_TO_NOT_CONTAMINATE = ["datetime","date","time","year","month","day","hour","minute","weekday","day_of_year","week_of_year","working_day","season","holiday","weekend","aemet_temperatura_media","aemet_precipitaciones","aemet_temperatura_minima","aemet_maxima_temperatura","aemet_direccion_media_viento","aemet_velocidad_media_viento","aemet_racha_maxima","aemet_humedad_media","aemet_humediad_maxima","aemet_humedad_minima","openweather_temperatura_media","openweather_punto_rocio","openweather_temperatura_sensacion","openweather_temperatura_minima","openweather_temperatura_maxima","openweather_presion_atmosferica","openweather_humedad_media","openweather_velocidad_viento_media","openweather_direccion_viento_media","openweather_racha_maxima_viento","openweather_lluvia_1h","openweather_cobertura_nubosa","openweather_identificador_temperatura","openweather_estado_temeperatura","openweather_descripcion_estado_temperatura","openweather_estado_temeperatura_name_Clouds","openweather_estado_temeperatura_name_Rain","openweather_descripcion_estado_temperatura_name_few-clouds","openweather_descripcion_estado_temperatura_name_heavy-intensity-rain","openweather_descripcion_estado_temperatura_name_light-rain","openweather_descripcion_estado_temperatura_name_moderate-rain","openweather_descripcion_estado_temperatura_name_overcast-clouds","openweather_descripcion_estado_temperatura_name_scattered-clouds","openweather_descripcion_estado_temperatura_name_sky-is-clear","openweather_descripcion_estado_temperatura_name_very-heavy-rain"]
# COLUMNAS QUE **NO DEBEN** SER CONTAMINADAS
# SE USAN PARA EXCLUIR CAMPOS COMO IDENTIFICADORES O FECHAS

ADD_LABEL = True
# TRUE = AÑADIR COLUMNA 'IS_ANOMALY' PARA INDICAR FILAS CONTAMINADAS
RANDOM_STATE = 42
# SEMILLA PARA ASEGURAR REPRODUCIBILIDAD DEL PROCESO ALEATORIO
SHOW_INFO = True
# TRUE = MOSTRAR INFORMACIÓN DE PROCESO EN CONSOLA

# CARGA DEL DATASET
df = pd.read_csv(INPUT_CSV)
if SHOW_INFO:
    print(f"[ INFO ] DATASET CARGADO: {df.shape[0]} FILAS, {df.shape[1]} COLUMNAS")


# SELECCIÓN DE COLUMNAS A CONTAMINAR
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()  # COLUMNAS NUMÉRICAS
cols_to_contaminate = [c for c in numeric_cols if c not in COLUMNS_TO_NOT_CONTAMINATE]  # EXCLUIR LAS DEFINIDAS

ignored_cols = [c for c in COLUMNS_TO_NOT_CONTAMINATE if c in df.columns]  # COLUMNAS EXCLUIDAS EXISTENTES

if not cols_to_contaminate:
    raise ValueError("[ ERROR ] NO HAY COLUMNAS VÁLIDAS PARA CONTAMINAR")

if SHOW_INFO:
    print(f"[ INFO ] COLUMNAS A CONTAMINAR: {len(cols_to_contaminate)}")
    if ignored_cols:
        print(f"[ INFO ] COLUMNAS NO CONTAMINADAS")

# SELECCIÓN DE FILAS A CONTAMINAR
np.random.seed(RANDOM_STATE)                     # FIJAR SEMILLA PARA REPRODUCIBILIDAD
n_rows = df.shape[0]                             # TOTAL DE FILAS
n_contam = int(n_rows * CONTAMINATION_RATE)      # CANTIDAD DE FILAS A CONTAMINAR
contam_indices = np.random.choice(df.index, size=n_contam, replace=False)  # ÍNDICES ALEATORIOS

if SHOW_INFO:
    print(f"[ INFO ] FILAS A CONTAMINAR: {n_contam}")

# COPIA DEL DATASET ORIGINAL
df_contaminated = df.copy()  # SE TRABAJA SOBRE UNA COPIA PARA NO ALTERAR EL ORIGINAL


# APLICAR CONTAMINACIÓN SEGÚN OPERACIÓN DEFINIDA
for col in cols_to_contaminate:
    if df[col].dtype in ['int64', 'float64']:  # SOLO COLUMNAS NUMÉRICAS
        # GENERAR RUIDO GAUSSIANO BASADO EN LA DESVIACIÓN ESTÁNDAR DE LA COLUMNA
        noise = np.random.normal(0, NOISE_INTENSITY * df[col].std(), size=n_contam)

        # APLICAR RUIDO SEGÚN LA OPERACIÓN DEFINIDA
        if OPERACION_RUIDO == 'SUMA':
            df_contaminated.loc[contam_indices, col] += noise  # SUMAR RUIDO
        elif OPERACION_RUIDO == 'RESTA':
            df_contaminated.loc[contam_indices, col] -= noise  # RESTAR RUIDO
        elif OPERACION_RUIDO == 'MULTIPLICACION':
            df_contaminated.loc[contam_indices, col] *= (1 + noise)  # ESCALAR VALOR ORIGINAL
        elif OPERACION_RUIDO == 'ESCALA':
            df_contaminated.loc[contam_indices, col] = df[col].mean() + noise  # MEDIA + RUIDO
        else:
            raise ValueError("[ ERROR ] OPERACION_RUIDO NO RECONOCIDA")

if SHOW_INFO:
    print(f"[ INFO ] CONTAMINACIÓN APLICADA CON OPERACIÓN '{OPERACION_RUIDO}'")

# AÑADIR COLUMNA DE ANOMALÍA
if ADD_LABEL:
    df_contaminated['is_anomaly'] = 0                        # TODAS FILAS NORMALES POR DEFECTO
    df_contaminated.loc[contam_indices, 'is_anomaly'] = 1    # MARCAR FILAS CONTAMINADAS
    if SHOW_INFO:
        print("[ INFO ] COLUMNA 'IS_ANOMALY' AÑADIDA")

# GUARDAR DATASET FINAL
df_contaminated.to_csv(OUTPUT_CSV, index=False)
if SHOW_INFO:
    print(f"[ GUARDADO ] DATASET CONTAMINADO EN '{OUTPUT_CSV}'")
