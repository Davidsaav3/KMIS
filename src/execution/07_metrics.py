import pandas as pd                            
import glob                                        
import os                                         
import json                                   
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, matthews_corrcoef  

# PARÁMETROS GENERALES
GLOBAL_FILE_PATTERN = '../../results/execution/06_global.csv'            # PATRÓN DEL CSV GLOBAL
OUTPUT_CSV = '../../results/execution/07_results.csv'  # CSV FINAL CON RESULTADOS
SHOW_INFO = True                                 # MOSTRAR INFO EN CONSOLA

# ORDEN DE COLUMNAS PARA CSV FINAL
columns_order = [
    'file', 'anomalies_real', 'anomalies_detected', 'detections_correct', 'false_positives', 'false_negatives',
    'precision', 'recall', 'f1_score', 'accuracy', 'mcc',
    'ratio_detection', 'ratio_fp', 'perc_global_anomalies_detected', 'perc_cluster_vs_global', 'total_coincidences'
]

df_global = pd.read_csv(GLOBAL_FILE_PATTERN)                             # LEER CSV GLOBAL
if 'anomaly' not in df_global.columns:
    raise ValueError("[ ERROR ] No se encontró columna 'anomaly' en IF global")

# VARIABLES GLOBALES
y_true_global = df_global['anomaly']                                  # ANOMALÍAS REALES GLOBALES
total_global = int(y_true_global.sum())
y_pred_global = y_true_global                                         # PREDICCIÓN PERFECTA PARA IF GLOBAL

# CALCULAR TP, FP, FN GLOBALES
tp_global = ((y_true_global==1) & (y_pred_global==1)).sum()           # VERDADEROS POSITIVOS
fp_global = ((y_true_global==0) & (y_pred_global==1)).sum()           # FALSOS POSITIVOS
fn_global = ((y_true_global==1) & (y_pred_global==0)).sum()           # FALSOS NEGATIVOS

# CREAR FILA DE RESULTADOS PARA IF GLOBAL
csv_rows = [{
    'file': 'global',                                               # NOMBRE DEL ARCHIVO
    'anomalies_real': int(y_true_global.sum()),                     # ANOMALÍAS REALES
    'anomalies_detected': int(y_pred_global.sum()),                 # ANOMALÍAS DETECTADAS
    'detections_correct': int(tp_global),                           # DETECCIONES CORRECTAS
    'false_positives': int(fp_global),                              # FALSOS POSITIVOS
    'false_negatives': int(fn_global),                              # FALSOS NEGATIVOS
    'precision': round(precision_score(y_true_global, y_pred_global, zero_division=0),4),  # PRECISIÓN
    'recall': round(recall_score(y_true_global, y_pred_global, zero_division=0),4),       # RECALL
    'f1_score': round(f1_score(y_true_global, y_pred_global, zero_division=0),4),         # F1 SCORE
    'accuracy': round(accuracy_score(y_true_global, y_pred_global),4),                     # ACCURACY
    'mcc': round(matthews_corrcoef(y_true_global, y_pred_global),4),                      # MCC
    'ratio_detection': round(recall_score(y_true_global, y_pred_global, zero_division=0),4),  # RATIO DETECCIÓN
    'ratio_fp': round(fp_global / len(y_true_global),4) if len(y_true_global)>0 else 0,    # RATIO FALSOS POSITIVOS
    'perc_global_anomalies_detected': 100.0,                                     # % GLOBAL DETECTADO
    'perc_cluster_vs_global': 100.0,                                            # % CLUSTER VS GLOBAL
    'total_coincidences': tp_global                                               # COINCIDENCIAS TOTALES
}]

# CREAR DATAFRAME FINAL Y GUARDAR CSV
df_csv = pd.DataFrame(csv_rows)[columns_order]                              # CONSTRUIR DATAFRAME FINAL
df_csv.to_csv(OUTPUT_CSV, index=False)                                       # GUARDAR CSV
if SHOW_INFO:
    print(f"[ GUARDADO ] CSV resumen de métricas en '{OUTPUT_CSV}'")         # CONFIRMACIÓN
