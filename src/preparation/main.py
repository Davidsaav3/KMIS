import subprocess  # PARA EJECUTAR OTROS SCRIPTS DESDE PYTHON
import sys          # PARA USAR EL INTERPRETE PYTHON ACTUAL
import logging      # PARA REGISTRAR MENSAJES DE INFORMACIÓN Y ERRORES
import os           # PARA MANEJO DE DIRECTORIOS Y RUTAS

# CONFIGURACIÓN GENERAL DE EJECUCIÓN
RESULTS_DIR = '../../results/preparation'   # CARPETA PRINCIPAL DONDE SE GUARDAN TODOS LOS LOGS Y RESULTADOS
LOG_FILE = os.path.join(RESULTS_DIR, 'log.txt')  # ARCHIVO DONDE SE GUARDAN TODOS LOS MENSAJES DEL LOG
OVERWRITE_LOG = True   # TRUE = SOBRESCRIBIR LOG EXISTENTE CADA VEZ, FALSE = AGREGAR NUEVOS MENSAJES AL FINAL
SHOW_STDOUT = True     # TRUE = MOSTRAR SALIDA ESTÁNDAR (PRINTS NORMALES) EN PANTALLA
SHOW_STDERR = True     # TRUE = MOSTRAR ERRORES (EXCEPCIONES Y MENSAJES DE ERROR) EN PANTALLA

# LISTA DE SCRIPTS A EJECUTAR CON OPCIÓN DE ACTIVACIÓN INDIVIDUAL
SCRIPTS = [
    {'name': '01_metrics.py', 'active': True},        # CALCULA ESTADÍSTICAS Y MÉTRICAS INICIALES DEL DATASET
    {'name': '02_nulls.py', 'active': True},          # PROCESA VALORES NULOS, LOS IMPUTA O ELIMINA
    {'name': '03_codification.py', 'active': True},   # CODIFICA VARIABLES CATEGÓRICAS (ONE-HOT O LABEL ENCODING)
    {'name': '04_scale.py', 'active': True},          # ESCALA DATOS NUMÉRICOS (STANDARD, MINMAX, ETC.)
    {'name': '05_variance.py', 'active': True},       # ELIMINA COLUMNAS CON VARIANZA BAJA O CONSTANTE
]

# CREAR CARPETA DE RESULTADOS SI NO EXISTE
# ASEGURA QUE TODOS LOS ARCHIVOS Y LOGS PUEDAN GUARDARSE SIN ERROR
os.makedirs(RESULTS_DIR, exist_ok=True)

# CONFIGURAR LOG
logging.basicConfig(
    filename=LOG_FILE,                              # RUTA DEL ARCHIVO DONDE SE GUARDAN LOS LOGS
    filemode='w' if OVERWRITE_LOG else 'a',         # SOBRESCRIBIR O AGREGAR DEPENDIENDO DE CONFIGURACIÓN
    level=logging.INFO,                             # NIVEL DE LOG: INFO PARA MENSAJES NORMALES
    format='%(message)s',                           # SOLO GUARDAR MENSAJE, SIN FECHA NI NIVEL
    encoding='utf-8'                                # SOPORTE PARA CARACTERES ESPECIALES COMO Ñ O ACENTOS
)

# FUNCIÓN AUXILIAR PARA IMPRIMIR MENSAJES EN PANTALLA Y GUARDAR EN LOG
def log_print(msg, level='info'):
    """IMPRIME MENSAJE EN PANTALLA Y LO GUARDA EN LOG SEGÚN NIVEL"""
    if level == 'info':
        logging.info(msg)     # REGISTRAR MENSAJE NORMAL
        if SHOW_STDOUT:       # OPCIONALMENTE IMPRIMIR EN PANTALLA
            print(msg)
    elif level == 'error':
        logging.error(msg)    # REGISTRAR MENSAJE DE ERROR
        if SHOW_STDERR:       # OPCIONALMENTE IMPRIMIR EN PANTALLA
            print(msg)

# MENSAJE INICIAL DE EJECUCIÓN
log_print("\n[ INICIO DE EJECUCIÓN DE SCRIPTS ]")

# BUCLE PRINCIPAL PARA EJECUTAR CADA SCRIPT
for script in SCRIPTS:
    if not script['active']:
        # OMITIR SCRIPT SI ESTÁ DESACTIVADO
        log_print(f"[ SKIP ] {script['name']} DESACTIVADO")
        continue

    log_print(f"\n[ EJECUTANDO ] {script['name']}\n")
    try:
        # EJECUTAR SCRIPT USANDO EL INTERPRETE ACTUAL
        process = subprocess.Popen(
            [sys.executable, script['name']],  # INTERPRETE + NOMBRE DE SCRIPT
            stdout=subprocess.PIPE,            # CAPTURAR SALIDA ESTÁNDAR
            stderr=subprocess.PIPE,            # CAPTURAR ERRORES
            text=True,                         # CAPTURAR SALIDA COMO TEXTO
            bufsize=1,                          # LECTURA LÍNEA A LÍNEA
            universal_newlines=True             # SOPORTE PARA NUEVAS LÍNEAS UNIVERSALES
        )

        # LEER SALIDA STDOUT EN TIEMPO REAL
        for line in process.stdout:
            log_print(line.rstrip())           # ELIMINA ESPACIOS/ENTER AL FINAL

        # LEER ERRORES STDERR EN TIEMPO REAL
        for line in process.stderr:
            log_print(line.rstrip(), level='error')

        # ESPERAR A QUE EL SCRIPT TERMINE
        process.wait()
        if process.returncode != 0:
            # SI EL CÓDIGO DE RETORNO ES DISTINTO DE 0, HUBO UN ERROR
            log_print(f"[ ERROR ] {script['name']} TERMINÓ CON CÓDIGO {process.returncode}", level='error')

    except Exception as e:
        # CAPTURAR CUALQUIER EXCEPCIÓN DURANTE LA EJECUCIÓN DEL SCRIPT
        log_print(f"[ EXCEPCIÓN ] {script['name']}: {e}", level='error')

# MENSAJE FINAL DE TERMINACIÓN
log_print("\n[ FIN ]")
