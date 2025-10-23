import pandas as pd                  # IMPORTA PANDAS PARA MANEJO DE DATOS
import matplotlib.pyplot as plt      # IMPORTA MATPLOTLIB PARA CREAR GRÁFICAS
import seaborn as sns                # IMPORTA SEABORN PARA ESTILO Y VISUALIZACIÓN AVANZADA
import os                            # IMPORTA OS PARA GESTIONAR RUTAS Y CARPETAS
import glob                          # IMPORTA GLOB PARA BUSCAR ARCHIVOS CON PATRONES
import numpy as np

# DEFINE LAS RUTAS Y PARÁMETROS GENERALES DEL SCRIPT
GLOBAL_CSV = '../../results/execution/04_global.csv'        # ARCHIVO GLOBAL CON RESULTADOS DEL ISOLATION FOREST
IF_01_CSV = '../../results/execution/04_global_if.csv'                # ARCHIVO CON SECUENCIAS DE ANOMALÍAS DETECTADAS
CLUSTER_FILES_PATH = '../../results/execution/cluster_*.csv'   # PATRÓN DE BÚSQUEDA DE ARCHIVOS POR CLUSTER
RESULTS_SUMMARY_CSV = '../../results/execution/06_results.csv' # ARCHIVO CON LAS MÉTRICAS DE RENDIMIENTO
RESULTS_FOLDER = '../../results/execution/plots'               # CARPETA DONDE SE GUARDARÁN LOS GRÁFICOS

FEATURE_TO_PLOT = 'nivel_plaxiquet'                           # VARIABLE PRINCIPAL A VISUALIZAR EN LOS GRÁFICOS
SAVE_FIGURES = True                                            # DEFINE SI SE GUARDAN LAS FIGURAS EN ARCHIVOS
SHOW_FIGURES = False                                           # DEFINE SI SE MUESTRAN EN PANTALLA LAS FIGURAS
STYLE = 'whitegrid'                                            # ESTILO VISUAL PARA SEABORN

# CREA LA CARPETA DE RESULTADOS SI NO EXISTE
os.makedirs(RESULTS_FOLDER, exist_ok=True)                     # CREA LA CARPETA SI NO EXISTE
print(f"[ INFO ] CARPETA '{RESULTS_FOLDER}' CREADA SI NO EXISTÍA")  # IMPRIME CONFIRMACIÓN EN CONSOLA

# CONFIGURA EL ESTILO Y EL TAMAÑO DE LAS FIGURAS
sns.set_style(STYLE)                                           # APLICA ESTILO VISUAL A SEABORN
plt.rcParams['figure.figsize'] = (12, 6)                       # DEFINE TAMAÑO POR DEFECTO DE LAS FIGURAS

# CARGA EL ARCHIVO PRINCIPAL CON LOS RESULTADOS DEL MODELO GLOBAL
df_if = pd.read_csv(GLOBAL_CSV)                             # LEE EL ARCHIVO CSV PRINCIPAL

# CONVIERTE VARIABLES A TIPO ENTERO PARA ASEGURAR CONSISTENCIA
df_if['anomaly'] = df_if['anomaly'].astype(int)                # CONVIERTE LA COLUMNA 'anomaly' A ENTERO
df_if['is_anomaly'] = df_if['is_anomaly'].astype(int)          # CONVIERTE LA COLUMNA 'is_anomaly' A ENTERO
df_if['sequence'] = df_if['sequence'].astype(int)              # CONVIERTE LA COLUMNA 'sequence' A ENTERO

# AÑADE LA COLUMNA CLUSTER SI NO EXISTE EN EL DATAFRAME
if 'cluster' not in df_if.columns:                             # VERIFICA SI FALTA LA COLUMNA
    df_if['cluster'] = 0                                       # CREA COLUMNA 'cluster' CON VALOR 0

# 1. GRÁFICO: ANOMALÍAS DETECTADAS VS REALES
plt.figure(figsize=(18, 6))                                    # CREA UNA NUEVA FIGURA DE TAMAÑO GRANDE
sns.scatterplot(                                               
    data=df_if,                                                 # USA EL DATAFRAME GLOBAL
    x='datetime',                                               # EJE X: FECHA Y HORA
    y=FEATURE_TO_PLOT,                                          # EJE Y: VARIABLE PRINCIPAL
    hue='anomaly',                                              # COLOR SEGÚN SI ES ANOMALÍA O NO
    palette={0: 'blue', 1: 'red'},                              # DEFINE COLORES (NORMAL=GRIS, ANOMALÍA=ROJO)
    alpha=0.7                                                   # DEFINE TRANSPARENCIA
)
plt.title(f"Anomalies Detected vs Real: {FEATURE_TO_PLOT.upper()}")  # TÍTULO DEL GRÁFICO
plt.xlabel("Datetime")                                          # ETIQUETA DEL EJE X
plt.ylabel(FEATURE_TO_PLOT.upper())                             # ETIQUETA DEL EJE Y
plt.xticks(rotation=45)                                         # ROTA ETIQUETAS DE TIEMPO
plt.legend(title='Anomaly', labels=['Real', 'Detected'])      # DEFINE LEYENDA
if SAVE_FIGURES:                                                # SI SE DEBE GUARDAR LA FIGURA
    plt.savefig(f"{RESULTS_FOLDER}/01_anomalies_vs_real.png", dpi=300, bbox_inches='tight')  # GUARDA EL GRÁFICO
plt.close()                                                     # CIERRA LA FIGURA PARA LIBERAR MEMORIA

# 2. HEATMAP DE CORRELACIÓN ENTRE VARIABLES NUMÉRICAS
numeric_cols = df_if.select_dtypes(include=['float64', 'int64']).columns  # SELECCIONA SOLO COLUMNAS NUMÉRICAS
plt.figure(figsize=(30, 18))                                  # CREA UNA FIGURA DE GRAN TAMAÑO
sns.heatmap(                                                  # CREA UN HEATMAP DE CORRELACIONES
    df_if[numeric_cols].corr(),                               # CALCULA MATRIZ DE CORRELACIÓN
    annot=True, fmt=".2f", cmap='coolwarm',                   # MUESTRA VALORES NUMÉRICOS CON FORMATO
    cbar=True, annot_kws={"size": 3}, linewidths=0.3, linecolor='white'  # CONFIGURA DETALLES VISUALES
)
plt.title("Correlation Matrix - All Numeric Features", fontsize=14)  # TÍTULO DEL HEATMAP
plt.xticks(rotation=45, ha='right', fontsize=4)               # ROTA Y AJUSTA ETIQUETAS X
plt.yticks(rotation=0, fontsize=4)                            # AJUSTA ETIQUETAS Y
if SAVE_FIGURES:
    plt.savefig(f"{RESULTS_FOLDER}/02_correlation_matrix.png", dpi=300, bbox_inches='tight')  # GUARDA EL HEATMAP
plt.close()

# 3. HISTOGRAMA DE SCORES DE ANOMALÍAS
df_if = pd.read_csv(IF_01_CSV)                             # LEE EL ARCHIVO CSV PRINCIPAL

plt.figure()                                                 # CREA NUEVA FIGURA
sns.histplot(df_if['anomaly_score'], bins=50, kde=True, color='red')  # CREA HISTOGRAMA CON CURVA KDE
plt.title("Distribution of Anomaly Scores")                  # TÍTULO DEL GRÁFICO
plt.xlabel("Anomaly Score")                                  # ETIQUETA EJE X
plt.ylabel("Frequency")                                      # ETIQUETA EJE Y
if SAVE_FIGURES:
    plt.savefig(f"{RESULTS_FOLDER}/03_anomaly_score_distribution.png", dpi=300)  # GUARDA EL HISTOGRAMA
plt.close()
# Parámetros
RESULTS_FOLDER = '../../results/execution/plots'
SAVE_FIGURES = True

# 4. Cargar archivo del cluster
# 4. Cargar archivo del cluster
df_cluster = pd.read_csv('../../results/execution/cluster_ubicacion_Ull_Pueblo.csv')

# Asegurar columna 'sequence'
if 'sequence' not in df_cluster.columns:
    df_cluster['sequence'] = 0

# Filtrar secuencias activas (>0) o usar todas si no hay ninguna
if df_cluster['sequence'].sum() > 0:
    df_sequences = df_cluster[df_cluster['sequence'] > 0].copy()
else:
    df_sequences = df_cluster.copy()

# Agrupar por 'datetime' (texto) y sumar secuencias
df_plot = df_sequences.groupby('datetime', as_index=False)['sequence'].sum()

# Si no hay datos, rellenar con 0 para evitar gráfico vacío
if df_plot.empty:
    df_plot = pd.DataFrame({
        'datetime': ['no_data'],
        'sequence': [0]
    })

# Convertir datetime a categoría ordenada
df_plot['datetime_cat'] = pd.Categorical(df_plot['datetime'], categories=df_plot['datetime'], ordered=True)

# Crear gráfico de barras ancho
plt.figure(figsize=(24, 6))
sns.barplot(
    data=df_plot,
    x='datetime_cat',
    y='sequence',
    color='skyblue'
)

plt.title("Temporal Distribution of Repeated Anomalies (Sequence Length)")
plt.xlabel("Datetime")
plt.ylabel("Total Anomaly Sequence Length")

# Reducir número de etiquetas X para que no colapse
max_labels = 20
step = max(1, len(df_plot) // max_labels)
plt.xticks(rotation=45)
plt.xticks(ticks=range(0, len(df_plot), step), labels=df_plot['datetime'][::step])

# Guardar figura
if SAVE_FIGURES:
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    plt.savefig(f"{RESULTS_FOLDER}/04_anomaly_sequence_over_time.png", dpi=300, bbox_inches='tight')
plt.close()

print("[ INFO ] Gráfico 4 guardado para el cluster")



CLUSTER_FILES_PATH = '../../results/execution/cluster_*.csv'
RESULTS_FOLDER = '../../results/execution/plots'
SAVE_FIGURES = True
target_cluster_name = "cluster_ubicacion_Ull_Pueblo.csv"  # Cluster específico

# Buscar archivo del cluster
cluster_files = glob.glob(CLUSTER_FILES_PATH)
selected_files = [f for f in cluster_files if target_cluster_name in os.path.basename(f)]

dfs_clusters = []    
# 5 y 6 CARGA Y COMBINA LOS ARCHIVOS DE CLUSTERS
for file in selected_files:                                    # RECORRE CADA ARCHIVO ENCONTRADO
    if not file.endswith('_if.csv'):                          # IGNORA LOS ARCHIVOS DE RESULTADOS IF
        dfc = pd.read_csv(file)                               # LEE EL ARCHIVO CSV
        if 'datetime' not in dfc.columns and 'timestamp' in dfc.columns:
            dfc['datetime'] = dfc['timestamp']                # CREA COLUMNA DATETIME SI NO EXISTE
        if 'cluster' not in dfc.columns:
            cluster_name = "cluster_" + os.path.splitext(os.path.basename(file))[0]
            dfc['cluster'] = cluster_name                     # AÑADE COLUMNA CLUSTER CON NOMBRE DINÁMICO
        dfs_clusters.append(dfc)                              # AÑADE EL DATAFRAME A LA LISTA

# CONCATENA TODOS LOS DATAFRAMES EN UNO SOLO
df_all_clusters = pd.concat(dfs_clusters, ignore_index=True)
df_if = pd.read_csv(GLOBAL_CSV)
 
# 5. SCATTER DETALLADO CON LEYENDA
plt.figure(figsize=(12, 6))
sns.scatterplot(
    data=df_all_clusters,
    x='sequence',                        
    y='anomaly_score',                   
    hue='cluster',                       
    palette='tab10',                     
    alpha=0.6,                           
    size='sequence',                     
    sizes=(20, 200)                      
)
plt.title("Anomaly Score vs Sequence Length by Cluster")
plt.xlabel("Sequence Length")
plt.ylabel("Anomaly Score")
plt.legend(title='Cluster', loc='upper right')
if SAVE_FIGURES:
    plt.savefig(f"{RESULTS_FOLDER}/05_sequence_vs_score_by_cluster.png", dpi=300, bbox_inches='tight')
plt.close()

# 6. DISTRIBUCIÓN TEMPORAL DE SCORES POR CLUSTER
plt.figure(figsize=(18, 6))
sns.scatterplot(
    data=df_all_clusters,
    x='datetime',                         
    y='anomaly_score',                    
    hue='cluster',                        
    palette='tab10',                      
    size='sequence',                     
    sizes=(20, 200),
    alpha=0.6                             
)
plt.title("Temporal Distribution of Anomaly Scores per Cluster")
plt.xlabel("Datetime")
plt.ylabel("Anomaly Score")
plt.xticks(rotation=45)
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
if SAVE_FIGURES:
    plt.savefig(f"{RESULTS_FOLDER}/06_temporal_score_by_cluster.png", dpi=300, bbox_inches='tight')
plt.close()

# CARGA MÉTRICAS DE RENDIMIENTO
df_summary = pd.read_csv(RESULTS_SUMMARY_CSV)          # Carga el CSV con las métricas calculadas de los modelos
df_summary.set_index('file', inplace=True)             # Usa el nombre del archivo (método o modelo) como índice

# 7. GRÁFICO DE MÉTRICAS DE RENDIMIENTO
metrics = ['precision', 'recall', 'f1_score', 'accuracy', 'mcc']  # Lista de métricas a visualizar
df_summary[metrics].plot(kind='bar', figsize=(16, 6))              # Gráfico de barras comparando el rendimiento por método
plt.title("Performance Metrics per Method / File")                 # Título del gráfico
plt.ylabel("Score")                                                # Etiqueta eje Y
plt.ylim(0, 1.1)                                                   # Escala del eje Y de 0 a 1.1 para ver bien las diferencias
plt.xticks(rotation=45, ha='right')                                # Rota las etiquetas X
plt.legend(title='Metric')                                         # Muestra la leyenda de métricas
plt.tight_layout()                                                 # Ajusta márgenes
if SAVE_FIGURES:
    plt.savefig(f"{RESULTS_FOLDER}/07_metrics_comparison.png", dpi=300)  # Guarda la figura
plt.close()

# 8. RATIO DE DETECCIÓN VS FALSOS POSITIVOS
ratio_metrics = ['anomalies_real','anomalies_detected','detections_correct', 'total_coincidences']                    # Selecciona las métricas de ratio
df_summary[ratio_metrics].plot(kind='bar',figsize=(16, 6),)
plt.title("Ratio Detection vs False Positives")                    # Título del gráfico
plt.ylabel("Ratio")                                                # Etiqueta eje Y
plt.xticks(rotation=45, ha='right')                                # Rota etiquetas X
plt.tight_layout()
if SAVE_FIGURES:
    plt.savefig(f"{RESULTS_FOLDER}/08_ratio_detection_fp.png", dpi=300)  # Guarda la figura
plt.close()

# 9. TRUE POSITIVES, FALSE POSITIVES, FALSE NEGATIVES
df_summary[['detections_correct', 'false_positives', 'false_negatives']].plot(
    kind='bar',
    figsize=(16, 6),
    color=['blue', 'green', 'red']  # Verde = correct detections, Rojo = FP, Naranja = FN
)
plt.title("True Positives, False Positives and False Negatives")
plt.ylabel("Count")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
if SAVE_FIGURES:
    plt.savefig(f"{RESULTS_FOLDER}/09_tp_fp_fn.png", dpi=300)
plt.close()

df_if = pd.read_csv('../../results/execution/cluster_ubicacion_Beniopa.csv')                             # LEE EL ARCHIVO CSV PRINCIPAL
# 10. GRÁFICO TEMPORAL DE ANOMALY SCORE POR CLUSTER Y ANOMALÍAS
# Asegurar tipos correctos
df_if['anomaly'] = df_if['anomaly'].astype(int)
df_if['anomaly_score'] = df_if['anomaly_score'].astype(float)

# Si no existe cluster, crear uno
if 'cluster' not in df_if.columns:
    df_if['cluster'] = 0

# Crear un color distinto para cada cluster
clusters = sorted(df_if['cluster'].unique())
palette = sns.color_palette("tab20", n_colors=len(clusters))  # Hasta 20 colores distintos
cluster_colors = {c: palette[i] for i, c in enumerate(clusters)}

plt.figure(figsize=(18, 6))

# Scatter plot con colores explícitos para cada cluster
for c in clusters:
    subset = df_if[df_if['cluster'] == c]
    plt.scatter(
        subset['datetime'],
        subset['anomaly_score'],
        color=cluster_colors[c],
        alpha=0.7,
        label=f'Cluster {c}'
    )

# Resaltar anomalías reales
anomalies = df_if[df_if['anomaly'] == 1]
plt.scatter(
    anomalies['datetime'],
    anomalies['anomaly_score'],
    color='red',
    s=100,
    edgecolor='black',
    label='Real Anomaly'
)

plt.title("Temporal Anomaly Scores by Cluster with Real Anomalies Highlighted")
plt.xlabel("Datetime")
plt.ylabel("Anomaly Score")
plt.xticks(rotation=45)
plt.legend(title='Cluster / Anomaly', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

if SAVE_FIGURES:
    plt.savefig(f"{RESULTS_FOLDER}/10_temporal_score_by_cluster_with_anomalies.png", dpi=300, bbox_inches='tight')
plt.close()

print("[ INFO ] Gráfico 10 guardado: Temporal Anomaly Score por Cluster con anomalías resaltadas")

# MENSAJE FINAL DE CONFIRMACIÓN
print("Visualizations saved in:", RESULTS_FOLDER)  # Mensaje en consola confirmando la ubicación de las gráficas

