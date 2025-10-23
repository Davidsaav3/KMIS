import pandas as pd  # PARA MANEJO DE DATAFRAMES
from sklearn.impute import SimpleImputer  # PARA CORREGIR VALORES NULOS
import json  # PARA GUARDAR INFORMACIÓN DE COLUMNAS EN FORMATO JSON

# PARÁMETROS 
INPUT_FILE = '../../data/dataset.csv'                  # RUTA DEL DATASET ORIGINAL
OUTPUT_CSV = '../../results/preparation/02_nulls.csv'  # RUTA DEL DATASET FINAL SIN NULOS
OUTPUT_JSON = '../../results/preparation/02_aux.json'  # RUTA DEL JSON CON INFORMACIÓN DE COLUMNAS

NUMERIC_STRATEGY = 'median'                            
# ESTRATEGIA PARA RELLENAR NaN EN COLUMNAS NUMÉRICAS
# OPCIONES:
# - 'mean'       : REEMPLAZA CON LA MEDIA. IMPLICA QUE LOS DATOS SE AJUSTAN AL PROMEDIO, PERO PUEDE SER SENSIBLE A OUTLIERS.
# - 'median'     : REEMPLAZA CON LA MEDIANA. RESISTENTE A OUTLIERS Y MANTIENE LA DISTRIBUCIÓN CENTRAL.
# - 'constant'   : REEMPLAZA CON FILL_CONSTANT_NUMERIC. PUEDE INTRODUCIR SESGO SI EL VALOR NO REPRESENTA LA DISTRIBUCIÓN REAL.
# IMPLICA QUE NO QUEDARÁN NaN Y LOS MODELOS PODRÁN PROCESAR TODAS LAS FILAS

FILL_CONSTANT_NUMERIC = 0                               
# VALOR FIJO PARA COLUMNAS NUMÉRICAS SI strategy='constant'
# IMPLICA QUE TODOS LOS NaN SE CONVERTIRÁN EN 0, LO QUE PUEDE INTRODUCIR SESGO SI 0 NO ES REPRESENTATIVO

CATEGORICAL_STRATEGY = 'most_frequent'                 
# ESTRATEGIA PARA RELLENAR NaN EN COLUMNAS CATEGÓRICAS
# OPCIONES:
# - 'most_frequent' : REEMPLAZA CON LA CATEGORÍA MÁS FRECUENTE. PUEDE SESGAR HACIA ESA CATEGORÍA SI HAY MUCHOS NaN.
# - 'constant'      : REEMPLAZA CON FILL_CONSTANT_CATEGORICAL. TRATA LOS NaN COMO NUEVA CATEGORÍA SEPARADA.
# IMPLICA QUE NO HABRÁ NaN Y LAS COLUMNAS PODRÁN SER CODIFICADAS PARA MODELOS

FILL_CONSTANT_CATEGORICAL = 'unknown'                  
# VALOR FIJO PARA COLUMNAS CATEGÓRICAS SI strategy='constant'
# IMPLICA QUE TODOS LOS NaN SE CONVERTIRÁN EN 'UNKNOWN' Y SE TRATARÁN COMO NUEVA CATEGORÍA EN LOS MODELOS

REMOVE_EMPTY_COLUMNS = True                             # ELIMINAR COLUMNAS COMPLETAMENTE VACÍAS
SHOW_INFO = True                                       # MOSTRAR MENSAJES INFORMATIVOS EN PANTALLA
SAVE_INTERMEDIATE = True                              # GUARDAR CSV INTERMEDIO ANTES DE CORREGIR NULOS

# CARGAR DATASET
df = pd.read_csv(INPUT_FILE)  # LEER EL CSV DE ENTRADA

# ELIMINAR COLUMNAS 'Unnamed:' AUTOMÁTICAS (errores CSV)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

if SHOW_INFO:
    print(f"[ INFO ] Dataset cargado: {df.shape[0]} FILAS, {df.shape[1]} COLUMNAS")

# IDENTIFICAR COLUMNAS POR TIPO
num_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()  # DETECTAR COLUMNAS NUMÉRICAS
cat_cols = df.select_dtypes(include=['object','category']).columns.tolist()  # DETECTAR COLUMNAS CATEGÓRICAS
if SHOW_INFO:
    print(f"[ INFO ] Columnas numéricas: {len(num_cols)}, Columnas categóricas: {len(cat_cols)}")

# GUARDAR INFORMACIÓN INICIAL DE COLUMNAS
columns_info = {
    'categorical': cat_cols,                # LISTA DE COLUMNAS CATEGÓRICAS DETECTADAS
    'numeric_imputed': num_cols,            # COLUMNAS NUMÉRICAS QUE SE CORREGIRAN
    'categorical_imputed': cat_cols,        # COLUMNAS CATEGÓRICAS QUE SE CORREGIRAN
    'removed_empty_columns': []             # COLUMNAS VACÍAS QUE SE ELIMINARÁN
}

# ELIMINAR COLUMNAS COMPLETAMENTE VACÍAS (OPCIONAL)
if REMOVE_EMPTY_COLUMNS:
    empty_cols = df.columns[df.isna().all()].tolist()  # DETECTAR COLUMNAS VACÍAS
    df.drop(columns=empty_cols, inplace=True)          # ELIMINAR COLUMNAS VACÍAS
    columns_info['removed_empty_columns'] = empty_cols  # GUARDAR LISTA DE COLUMNAS ELIMINADAS
    if SHOW_INFO:
        print(f"[ INFO ] Columnas completamente vacías eliminadas: {len(empty_cols)}")

# GUARDAR DATASET INTERMEDIO (OPCIONAL)
if SAVE_INTERMEDIATE:
    intermediate_csv = OUTPUT_CSV.replace('.csv','_intermediate.csv')  # NOMBRE DEL CSV INTERMEDIO
    df.to_csv(intermediate_csv, index=False)  # GUARDAR DATASET INTERMEDIO
    if SHOW_INFO:
        print(f"[ GUARDADO ] Dataset intermedio en '{intermediate_csv}'")

# FILTRAR SOLO COLUMNAS EXISTENTES (NUMÉRICAS)
num_cols_actuales = [c for c in num_cols if c in df.columns]
if len(num_cols_actuales) < len(num_cols):
    faltantes = set(num_cols) - set(num_cols_actuales)
    print(f"[WARNING] Columnas numéricas no encontradas y se omiten: {faltantes}")

# CORREGIR COLUMNAS NUMÉRICAS
num_strategy_params = {'strategy': NUMERIC_STRATEGY}  # CONFIGURAR ESTRATEGIA
if NUMERIC_STRATEGY == 'constant':  # SI USAMOS CONSTANTE, DEFINIR VALOR
    num_strategy_params['fill_value'] = FILL_CONSTANT_NUMERIC

imputer_num = SimpleImputer(**num_strategy_params)  # CREAR OBJETO IMPUTER
if num_cols_actuales:  # SOLO IMPUTAR SI HAY COLUMNAS DISPONIBLES
    df[num_cols_actuales] = imputer_num.fit_transform(df[num_cols_actuales])  # CORREGIR NULOS NUMÉRICOS
    if SHOW_INFO:
        print(f"[ INFO ] Valores nulos numéricos corregidos con '{NUMERIC_STRATEGY}'")

# FILTRAR SOLO COLUMNAS EXISTENTES (CATEGÓRICAS)
cat_cols_actuales = [c for c in cat_cols if c in df.columns]
if len(cat_cols_actuales) < len(cat_cols):
    faltantes = set(cat_cols) - set(cat_cols_actuales)
    print(f"[WARNING] Columnas categóricas no encontradas y se omiten: {faltantes}")

# CORREGIR COLUMNAS CATEGÓRICAS
cat_strategy_params = {'strategy': CATEGORICAL_STRATEGY}  # CONFIGURAR ESTRATEGIA
if CATEGORICAL_STRATEGY == 'constant':  # SI USAMOS CONSTANTE, DEFINIR VALOR
    cat_strategy_params['fill_value'] = FILL_CONSTANT_CATEGORICAL

imputer_cat = SimpleImputer(**cat_strategy_params)  # CREAR OBJETO IMPUTER
if cat_cols_actuales:  # SOLO IMPUTAR SI HAY COLUMNAS DISPONIBLES
    df[cat_cols_actuales] = imputer_cat.fit_transform(df[cat_cols_actuales])  # CORREGIR NULOS CATEGÓRICOS
    if SHOW_INFO:
        print(f"[ INFO ] Valores nulos categóricos corregidos con '{CATEGORICAL_STRATEGY}'")

# GUARDAR DATASET FINAL
df.to_csv(OUTPUT_CSV, index=False)  # GUARDAR CSV FINAL SIN NULOS
if SHOW_INFO:
    print(f"[ GUARDADO ] Dataset final sin nulos en '{OUTPUT_CSV}'")
    print(f"[ INFO ] Dataset final: {df.shape[0]} FILAS, {df.shape[1]} COLUMNAS")

# GUARDAR INFORMACIÓN DE COLUMNAS EN JSON
with open(OUTPUT_JSON, 'w') as f:
    json.dump(columns_info, f, indent=4)  # GUARDAR COLUMNAS CORREGIDAS Y ELIMINADAS
if SHOW_INFO:
    print(f"[ GUARDADO ] Columnas CORREGIDAS guardadas en '{OUTPUT_JSON}'")
