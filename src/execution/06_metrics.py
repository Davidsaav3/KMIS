import pandas as pd                                # IMPORTAR PANDAS PARA MANEJO DE DATAFRAMES
import glob                                        # IMPORTAR GLOB PARA LISTAR ARCHIVOS CON PATRONES
import os                                          # IMPORTAR OS PARA MANEJO DE RUTAS Y CARPETAS
import json                                        # IMPORTAR JSON PARA CARGAR DEFINICIÓN DE CLUSTERS
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, matthews_corrcoef  # MÉTRICAS DE RENDIMIENTO

# PARÁMETROS
RESULTS_FOLDER = '../../results/execution'       # CARPETA DE RESULTADOS
EXECUTION_FOLDER = '../../results/execution'     # CARPETA DE EJECUCIÓN
GLOBAL_FILE_PATTERN = '04_global.csv'                # PATRÓN PARA ARCHIVO IF GLOBAL
CLUSTERS_JSON = 'clusters.json'                  # ARCHIVO JSON CON DEFINICIÓN DE CLUSTERS
OUTPUT_CSV = os.path.join(RESULTS_FOLDER, '06_results.csv')  # CSV FINAL CON RESULTADOS
SHOW_INFO = True                                 # MOSTRAR INFORMACIÓN EN CONSOLA

# ORDEN DE COLUMNAS PARA CSV FINAL
columns_order = [
    'file', 'anomalies_real', 'anomalies_detected', 'detections_correct', 'false_positives', 'false_negatives',
    'total_sequences', 'max_sequence', 'precision', 'recall', 'f1_score', 'accuracy', 'mcc',
    'ratio_detection', 'ratio_fp', 'perc_global_anomalies_detected', 'perc_cluster_vs_global', 'total_coincidences'
]

# CARGAR IF GLOBAL
files = glob.glob(os.path.join(EXECUTION_FOLDER, '*.csv'))           # LISTAR TODOS LOS CSV EN EJECUCIÓN
global_files = [f for f in files if GLOBAL_FILE_PATTERN in os.path.basename(f)]  # FILTRAR EL CSV GLOBAL
if not global_files:
    raise FileNotFoundError(f"[ ERROR ] No se encontró archivo con patrón '{GLOBAL_FILE_PATTERN}'")

df_global = pd.read_csv(global_files[0])                             # LEER CSV GLOBAL
if 'anomaly' not in df_global.columns:
    raise ValueError("[ ERROR ] No se encontró columna 'anomaly' en IF global")
if 'sequence' not in df_global.columns:
    raise ValueError("[ ERROR ] No existe columna 'sequence' en IF global")

def contar_secuencias(col):
    seq = col.fillna(0).reset_index(drop=True)
    n = len(seq)
    count = 0
    i = 0
    while i < n:
        if seq[i] == 0:
            i += 1
            continue
        # Detectar bloque no-cero
        j = i
        bloque = []
        while j < n and seq[j] != 0:
            bloque.append(seq[j])
            j += 1
        # Contar si el bloque tiene al menos una secuencia creciente consecutiva
        if len(bloque) >= 2 and all(bloque[k] == bloque[k-1] + 1 for k in range(1, len(bloque))):
            count += 1
        i = j  # saltar al siguiente bloque
    return count

# DEFINIR VARIABLES GLOBALES
y_true_global = df_global['anomaly']                                  # ANOMALÍAS REALES GLOBALES
total_global = int(y_true_global.sum())
total_seq= contar_secuencias(df_global['sequence'])                                                           # TOTAL DE ANOMALÍAS
max_sequence_global = int(df_global['sequence'].max())                # LONGITUD MÁXIMA DE SECUENCIA
y_pred_global = y_true_global                                         # IF GLOBAL SE CONSIDERA PREDICCIÓN PERFECTA

# CALCULAR TP, FP, FN GLOBALES
tp_global = ((y_true_global==1) & (y_pred_global==1)).sum()           # VERDADEROS POSITIVOS
fp_global = ((y_true_global==0) & (y_pred_global==1)).sum()           # FALSOS POSITIVOS
fn_global = ((y_true_global==1) & (y_pred_global==0)).sum()           # FALSOS NEGATIVOS

# CREAR FILA DE RESULTADOS DEL IF GLOBAL
csv_rows = [{
    'file': 'global',                                               # NOMBRE DEL ARCHIVO
    'anomalies_real': int(y_true_global.sum()),                         # ANOMALÍAS REALES
    'anomalies_detected': int(y_pred_global.sum()),                     # ANOMALÍAS DETECTADAS
    'detections_correct': int(tp_global),                               # DETECCIONES CORRECTAS
    'false_positives': int(fp_global),                                  # FALSOS POSITIVOS
    'false_negatives': int(fn_global),                                  # FALSOS NEGATIVOS
    'total_sequences': total_seq,                                     # TOTAL DE SECUENCIAS
    'max_sequence': max_sequence_global,                                 # LONGITUD MÁXIMA DE SECUENCIA
    'precision': round(precision_score(y_true_global, y_pred_global, zero_division=0),4),  # PRECISIÓN
    'recall': round(recall_score(y_true_global, y_pred_global, zero_division=0),4),       # RECALL
    'f1_score': round(f1_score(y_true_global, y_pred_global, zero_division=0),4),         # F1 SCORE
    'accuracy': round(accuracy_score(y_true_global, y_pred_global),4),                     # ACCURACY
    'mcc': round(matthews_corrcoef(y_true_global, y_pred_global),4),                      # MCC
    'ratio_detection': round(recall_score(y_true_global, y_pred_global, zero_division=0),4),  # RATIO DE DETECCIÓN
    'ratio_fp': round(fp_global / len(y_true_global),4) if len(y_true_global)>0 else 0,  # RATIO FALSOS POSITIVOS
    'perc_global_anomalies_detected': 100.0,                                    # PORCENTAJE GLOBAL DETECTADO
    'perc_cluster_vs_global': 100.0,                                           # % CLUSTER VS GLOBAL
    'total_coincidences': tp_global                                              # COINCIDENCIAS TOTALES
}]

# CARGAR DEFINICIÓN DE CLUSTERS
with open(CLUSTERS_JSON, 'r', encoding='utf-8') as f:
    clusters_json = json.load(f)
if SHOW_INFO:
    print(f"[ INFO ] Clusters cargados desde '{CLUSTERS_JSON}'")

# PROCESAR CADA SUBCLUSTER
for cluster_name, subclusters in clusters_json.items():                   # ITERAR SOBRE CLUSTERS
    for sub_name in subclusters.keys():                                    # ITERAR SOBRE SUBCLUSTERS
        file_path = os.path.join(EXECUTION_FOLDER, f"cluster_{cluster_name}_{sub_name}.csv")
        if not os.path.exists(file_path):                                   # SI NO EXISTE ARCHIVO
            csv_rows.append({'file': f"{cluster_name}_{sub_name}", 'error': 'archivo no encontrado'})
            continue

        df_sub = pd.read_csv(file_path)                                     # LEER CSV DE SUBCLUSTER
        if 'anomaly' not in df_sub.columns or 'sequence' not in df_sub.columns:  # VALIDAR COLUMNAS
            csv_rows.append({'file': f"{cluster_name}_{sub_name}", 'error': 'columna anomaly o sequence no encontrada'})
            continue

        total_seq= contar_secuencias(df_sub['sequence'])

        y_pred = df_sub['anomaly']                                          # PREDICCIONES DEL SUBCLUSTER
        tp = ((y_true_global==1) & (y_pred==1)).sum()                       # TP CON RESPECTO GLOBAL
        fp = ((y_true_global==0) & (y_pred==1)).sum()                       # FP CON RESPECTO GLOBAL
        fn = ((y_true_global==1) & (y_pred==0)).sum()                       # FN CON RESPECTO GLOBAL
        total_cluster_anomalies = int(y_pred.sum())                          # TOTAL DE ANOMALÍAS DEL CLUSTER

        perc_global_anomalies_detected = round(tp / total_global * 100,2) if total_global>0 else 0   # % ANOMALÍAS GLOBALES DETECTADAS
        perc_cluster_vs_global = round(tp / total_cluster_anomalies * 100,2) if total_cluster_anomalies>0 else 0  # % CLUSTER VS GLOBAL

        # AÑADIR FILA AL CSV FINAL
        csv_rows.append({
            'file': f"{cluster_name}_{sub_name}",
            'anomalies_real': int(y_true_global.sum()),
            'anomalies_detected': total_cluster_anomalies,
            'detections_correct': int(tp),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'total_sequences': total_seq,
            'max_sequence': int(df_sub['sequence'].max()),
            'precision': round(precision_score(y_true_global, y_pred, zero_division=0),4),
            'recall': round(recall_score(y_true_global, y_pred, zero_division=0),4),
            'f1_score': round(f1_score(y_true_global, y_pred, zero_division=0),4),
            'accuracy': round(accuracy_score(y_true_global, y_pred),4),
            'mcc': round(matthews_corrcoef(y_true_global, y_pred),4),
            'ratio_detection': round(recall_score(y_true_global, y_pred, zero_division=0),4),
            'ratio_fp': round(fp / len(y_true_global),4) if len(y_true_global)>0 else 0,
            'perc_global_anomalies_detected': perc_global_anomalies_detected,
            'perc_cluster_vs_global': perc_cluster_vs_global,
            'total_coincidences': int(tp)
        })

# CREAR DATAFRAME FINAL Y GUARDAR CSV
df_csv = pd.DataFrame(csv_rows)[columns_order]                              # CONSTRUIR DATAFRAME FINAL
df_csv.to_csv(OUTPUT_CSV, index=False)                                       # GUARDAR CSV
if SHOW_INFO:
    print(f"[ GUARDADO ] CSV resumen de métricas en '{OUTPUT_CSV}'")
