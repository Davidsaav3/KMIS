import pandas as pd  # PARA MANEJO DE DATAFRAMES
from sklearn.feature_selection import VarianceThreshold  # PARA ELIMINAR COLUMNAS CON BAJA VARIANZA
from sklearn.preprocessing import StandardScaler  # IMPORTADO PERO NO USADO AQUÍ, POSIBLEMENTE PARA ESCALADO FUTURO
import json  

# CONFIGURACIÓN DE ARCHIVOS Y PARÁMETROS
INPUT_CSV = '../../results/preparation/04_scale.csv'        # DATASET DE ENTRADA ESCALADO
OUTPUT_CSV = '../../results/preparation/05_variance.csv'    # DATASET FINAL TRAS ELIMINAR COLUMNAS CON VARIANZA BAJA
OUTPUT_JSON = '../../results/preparation/05_aux.json'       # JSON CON LISTA DE COLUMNAS SELECCIONADAS Y EXCLUIDAS

VAR_THRESHOLD = 0.01            # UMBRAL DE VARIANZA: COLUMNAS CON VAR < 0.01 SERÁN ELIMINADAS

DROP_CONSTANT_COLUMNS = True     # TRUE = ELIMINAR COLUMNAS CON VARIANZA EXACTAMENTE 0
DROP_LOW_VARIANCE = True         # TRUE = ELIMINAR COLUMNAS CON VARIANZA < VAR_THRESHOLD

NUMERIC_ONLY = True              # TRUE = SOLO SE ANALIZAN COLUMNAS NUMÉRICAS

# CARGAR DATASET
df = pd.read_csv(INPUT_CSV)  # LEER EL CSV DE ENTRADA
if NUMERIC_ONLY:
    # CONSERVAR SOLO COLUMNAS NUMÉRICAS PARA EL ANÁLISIS DE VARIANZA
    df = df.select_dtypes(include=['int64','float64'])
print(f"[ INFO ] DATASET CARGADO: {df.shape[0]} FILAS, {df.shape[1]} COLUMNAS NUMÉRICAS")

# SELECCIÓN DE COLUMNAS POR VARIANZA
if DROP_CONSTANT_COLUMNS or DROP_LOW_VARIANCE:
    # CREAR OBJETO VarianceThreshold CON EL UMBRAL DEFINIDO
    selector = VarianceThreshold(threshold=VAR_THRESHOLD)
    
    # APLICAR SELECCIÓN DE COLUMNAS CON VARIANZA >= VAR_THRESHOLD
    df_selected = pd.DataFrame(
        selector.fit_transform(df),  # TRANSFORMAR DATAFRAME ELIMINANDO COLUMNAS DE BAJA VARIANZA
        columns=df.columns[selector.get_support()]  # CONSERVAR LOS NOMBRES DE COLUMNAS SELECCIONADAS
    )
    
    # OBTENER COLUMNAS EXCLUIDAS (VAR < UMBRAL)
    excluded_cols = list(df.columns[~selector.get_support()])
    
    # INFORMACIÓN SOBRE RESULTADO DE LA SELECCIÓN
    print(f"[ INFO ] COLUMNAS SELECCIONADAS: {df_selected.shape[1]} / {df.shape[1]}")
    print(f"[ INFO ] COLUMNAS EXCLUIDAS: {len(excluded_cols)}")
else:
    # SI NO SE ELIMINA NINGUNA COLUMNA, COPIAR EL DATAFRAME ORIGINAL
    df_selected = df.copy()
    excluded_cols = []

# GUARDAR DATASET FINAL CON COLUMNAS SELECCIONADAS
df_selected.to_csv(OUTPUT_CSV, index=False)
print(f"[ GUARDADO ] DATASET FINAL EN '{OUTPUT_CSV}'")

# GUARDAR INFORMACIÓN DE COLUMNAS SELECCIONADAS Y EXCLUIDAS EN JSON
with open(OUTPUT_JSON, 'w') as f:
    json.dump({
        'selected_columns': df_selected.columns.tolist(),  # LISTA DE COLUMNAS MANTENIDAS
        'excluded_columns': excluded_cols                 # LISTA DE COLUMNAS ELIMINADAS
    }, f, indent=4)
print(f"[ GUARDADO ] INFORMACIÓN DE COLUMNAS EN JSON EN '{OUTPUT_JSON}'")

# MENSAJE FINAL
print("[ INFO ] PROCESO DE SELECCIÓN POR VARIANZA FINALIZADO CON ÉXITO")
