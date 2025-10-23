import subprocess 
import sys 
import logging 
import os  
import json 

# PARÁMETROS DE EJECUCIÓN
RESULTS_FOLDER = '../../results/execution'  # CARPETA DE RESULTADOS
LOG_FILE = os.path.join(RESULTS_FOLDER, 'log.txt')  # ARCHIVO DE LOG
LOG_LEVEL = logging.INFO  # NIVEL DE LOG
LOG_OVERWRITE = True  # SOBRESCRIBIR LOG EXISTENTE
SHOW_OUTPUT = True  # MOSTRAR SALIDA EN CONSOLA

# LISTA DE SCRIPTS A EJECUTAR EN ORDEN
SCRIPTS = [
    '00_contaminate.py',  # CONTAMINAR DATOS
    '01_sample_size.py',  # AJUSTE TAMAÑO MUESTRA
    '02_number_of_trees.py',  # AJUSTE NÚMERO DE ÁRBOLES
    '03_max_number_of_features.py',  # AJUSTE MÁXIMO DE FEATURES
    '04_max_depth.py',  # AJUSTE PROFUNDIDAD MÁXIMA
    '05_detection_threshold.py',  # AJUSTE UMBRAL DETECCIÓN
    '06_if.py',  # EJECUCIÓN DEL ALGORITMO IF
    '07_metrics.py',  # CÁLCULO DE MÉTRICAS
    '08_visualize.py'  # VISUALIZACIÓN
]

# CREAR CARPETA DE RESULTADOS SI NO EXISTE
os.makedirs(RESULTS_FOLDER, exist_ok=True)  # CREAR CARPETA RESULTADOS

# CONFIGURAR LOG
logging.basicConfig(
    filename=LOG_FILE,
    filemode='w' if LOG_OVERWRITE else 'a',  # MODO ESCRITURA O ADICIÓN
    level=LOG_LEVEL,
    format='%(message)s',
    encoding='utf-8'
)

# FUNCIÓN PARA LOGUEAR Y MOSTRAR EN PANTALLA
def log_print(msg, level='info'):
    if level == 'info':
        logging.info(msg)  # GUARDAR MENSAJE INFO
        if SHOW_OUTPUT:
            print(msg)  # MOSTRAR EN CONSOLA
    elif level == 'error':
        logging.error(msg)  # GUARDAR MENSAJE DE ERROR
        if SHOW_OUTPUT:
            print(msg)  # MOSTRAR ERROR EN CONSOLA

log_print("[ INICIO ]\n")  # INICIO DE EJECUCIÓN

# LEER HIPERPARÁMETROS INICIALES O CREAR POR DEFECTO
HIP_JSON = os.path.join(RESULTS_FOLDER, 'hiperparameters.json')
default_values = {}
if os.path.exists(HIP_JSON):
    with open(HIP_JSON, 'r', encoding='utf-8') as f:
        default_values = json.load(f)  # CARGAR JSON EXISTENTE
else:
    default_values = {  # VALORES POR DEFECTO
        'S': {'value': 256},  # TAMAÑO MUESTRA
        'T': {'value': 100},  # NÚMERO DE ÁRBOLES
        'F': {'value': 1.0},  # MÁXIMO DE FEATURES
        'D': {'value': 1},    # PROFUNDIDAD MÁXIMA
        'Th': {'value': 0.01} # UMBRAL DETECCIÓN
    }

# EJECUTAR CADA SCRIPT EN ORDEN
for script in SCRIPTS:
    log_print(f"\n[ EJECUTANDO ] {script}\n")
    try:
        process = subprocess.Popen(
            [sys.executable, script],  # EJECUTAR SCRIPT CON PYTHON
            stdout=subprocess.PIPE,     # CAPTURAR SALIDA ESTÁNDAR
            stderr=subprocess.PIPE,     # CAPTURAR ERRORES
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # LEER SALIDA ESTÁNDAR DEL SCRIPT
        for line in process.stdout:
            log_print(line.rstrip())  # IMPRIMIR CADA LÍNEA

        # LEER ERRORES DEL SCRIPT
        for line in process.stderr:
            log_print(line.rstrip(), level='error')  # IMPRIMIR ERRORES

        process.wait()  # ESPERAR A QUE TERMINE EL SCRIPT
        if process.returncode != 0:
            log_print(f"[ ERROR ] {script} TERMINÓ CON CÓDIGO {process.returncode}", level='error')

    except Exception as e:
        log_print(f"[ EXCEPCIÓN ] {script}: {e}", level='error')  # CAPTURAR EXCEPCIONES

# LEER VALORES AJUSTADOS FINALES DE HIPERPARÁMETROS
if os.path.exists(HIP_JSON):
    with open(HIP_JSON, 'r', encoding='utf-8') as f:
        final_values = json.load(f)  # CARGAR JSON FINAL
else:
    final_values = default_values.copy()  # USAR VALORES POR DEFECTO

# MOSTRAR RESUMEN FINAL DE HIPERPARÁMETROS
log_print("\n[ RESUMEN HIPERPARÁMETROS ]")
for param in ['S','T','F','D','Th']:
    default_val = default_values.get(param, {}).get('default', 'N/A')
    final_val = final_values.get(param, {}).get('value', 'N/A')
    log_print(f"{param}: POR DEFECTO = {default_val}   |   AJUSTADO = {final_val}")  # IMPRIMIR RESUMEN

log_print("\n[ FIN ]")  # FIN DE EJECUCIÓN
