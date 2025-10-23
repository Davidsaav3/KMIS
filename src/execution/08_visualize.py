import pandas as pd                  # IMPORTA PANDAS PARA MANEJO DE DATOS
import matplotlib.pyplot as plt      # IMPORTA MATPLOTLIB PARA CREAR GRÁFICAS
import seaborn as sns                # IMPORTA SEABORN PARA ESTILO Y VISUALIZACIÓN AVANZADA
import os                            # IMPORTA OS PARA GESTIONAR RUTAS Y CARPETAS
import glob                          # IMPORTA GLOB PARA BUSCAR ARCHIVOS CON PATRONES
import numpy as np                   # IMPORTA NUMPY PARA OPERACIONES NUMÉRICAS

# CONFIGURACIÓN GENERAL
GLOBAL_CSV = '../../results/execution/06_global.csv'        # CSV GLOBAL CON RESULTADOS DEL MODELO
RESULTS_SUMMARY_CSV = '../../results/execution/07_results.csv' # CSV CON MÉTRICAS DE RENDIMIENTO
RESULTS_FOLDER = '../../results/execution/plots'               # CARPETA PARA GUARDAR GRÁFICOS
FEATURE_TO_PLOT = 'nivel_plaxiquet'                           # VARIABLE PRINCIPAL PARA GRAFICAR
SAVE_FIGURES = True                                            # GUARDAR FIGURAS EN ARCHIVO
SHOW_FIGURES = False                                           # MOSTRAR FIGURAS EN PANTALLA
STYLE = 'whitegrid'                                            # ESTILO VISUAL PARA SEABORN

# CREAR CARPETA DE RESULTADOS SI NO EXISTE
os.makedirs(RESULTS_FOLDER, exist_ok=True)                     # CREA CARPETA SI NO EXISTE
print(f"[ INFO ] CARPETA '{RESULTS_FOLDER}' CREADA SI NO EXISTÍA")  # CONFIRMACIÓN

# CONFIGURAR ESTILO Y TAMAÑO DE FIGURAS
sns.set_style(STYLE)                                           # APLICA ESTILO SEABORN
plt.rcParams['figure.figsize'] = (12, 6)                       # TAMAÑO POR DEFECTO DE FIGURAS

# CARGAR DATOS PRINCIPALES
df_if = pd.read_csv(GLOBAL_CSV)                                # LEER CSV GLOBAL
df_if['anomaly'] = df_if['anomaly'].astype(int)                # ASEGURAR TIPO ENTERO
df_if['is_anomaly'] = df_if['is_anomaly'].astype(int)

# CARGAR MÉTRICAS DE RENDIMIENTO
df_summary = pd.read_csv(RESULTS_SUMMARY_CSV)
df_summary.set_index('file', inplace=True)  # USAR NOMBRE DE ARCHIVO COMO ÍNDICE

# 1. GRAFICO: ANOMALÍAS DETECTADAS VS REALES
plt.figure(figsize=(18, 6))
sns.scatterplot(
    data=df_if,
    x='datetime',
    y=FEATURE_TO_PLOT,
    hue='anomaly',
    palette={0: 'blue', 1: 'red'},
    alpha=0.7
)
plt.title(f"Anomalies Detected vs Real: {FEATURE_TO_PLOT.upper()}")
plt.xlabel("Datetime")
plt.ylabel(FEATURE_TO_PLOT.upper())
plt.xticks(rotation=45)
plt.legend(title='Anomaly', labels=['Real', 'Detected'])
if SAVE_FIGURES:
    plt.savefig(f"{RESULTS_FOLDER}/01_anomalies_vs_real.png", dpi=300, bbox_inches='tight')
plt.close()  # LIBERAR MEMORIA

# 2. GRÁFICO DE MÉTRICAS DE RENDIMIENTO
metrics = ['precision', 'recall', 'f1_score', 'accuracy', 'mcc']
df_summary[metrics].plot(kind='bar', figsize=(16, 6))
plt.title("Performance Metrics per Method / File")
plt.ylabel("Score")
plt.ylim(0, 1.1)
plt.xticks(rotation=45, ha='right')
plt.legend(title='Metric')
plt.tight_layout()
if SAVE_FIGURES:
    plt.savefig(f"{RESULTS_FOLDER}/02_metrics_comparison.png", dpi=300)
plt.close()

# 3. RATIO DE DETECCIÓN VS FALSOS POSITIVOS
ratio_metrics = ['anomalies_real','anomalies_detected','detections_correct', 'total_coincidences']
df_summary[ratio_metrics].plot(kind='bar', figsize=(16, 6))
plt.title("Ratio Detection vs False Positives")
plt.ylabel("Ratio")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
if SAVE_FIGURES:
    plt.savefig(f"{RESULTS_FOLDER}/03_ratio_detection_fp.png", dpi=300)
plt.close()

# 4. TRUE POSITIVES, FALSE POSITIVES, FALSE NEGATIVES
df_summary[['detections_correct', 'false_positives', 'false_negatives']].plot(
    kind='bar',
    figsize=(16, 6),
    color=['blue', 'green', 'red']
)
plt.title("True Positives, False Positives and False Negatives")
plt.ylabel("Count")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
if SAVE_FIGURES:
    plt.savefig(f"{RESULTS_FOLDER}/04_tp_fp_fn.png", dpi=300)
plt.close()

# MENSAJE FINAL
print("Visualizations saved in:", RESULTS_FOLDER)
