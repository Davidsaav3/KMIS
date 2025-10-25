import pandas as pd
import unicodedata

# --- CARGAR DATOS ---
df = pd.read_csv("insolacion.csv", sep=None, engine="python", encoding="utf-8-sig")  # elimina BOM

# --- LIMPIAR NOMBRES DE COLUMNAS ---
def normalizar_columna(col):
    col = col.strip()
    col = ''.join(c for c in unicodedata.normalize('NFD', col) if unicodedata.category(c) != 'Mn')
    return col.upper()

df.columns = [normalizar_columna(c) for c in df.columns]

# --- DETECTAR COLUMNAS DE FECHA ---
id_vars = [c for c in df.columns if c in ['YEAR', 'MONTH', 'DAY']]
if len(id_vars) < 3:
    print(f"[ERROR] No se encontraron todas las columnas de fecha. Detectadas: {id_vars}")
    print("[INFO] Columnas disponibles:", list(df.columns))
    raise SystemExit

# --- DETECTAR COLUMNAS CON hour ---
hour_cols = [col for col in df.columns if "_" in col]
if not hour_cols:
    raise ValueError("No se detectaron columnas con formato VARIABLE_HH (por ejemplo BJ_07).")

# --- DESPIVOTAR ---
df_melt = df.melt(id_vars=id_vars, value_vars=hour_cols,
                  var_name="variable_hour", value_name="valor")

# --- SEPARAR VARIABLE Y hour ---
df_melt[["variable", "hour"]] = df_melt["variable_hour"].str.split("_", expand=True)

# --- LIMPIAR Y REORDENAR ---
df_final = df_melt.drop(columns=["variable_hour"])
df_final = df_final[id_vars + ["hour", "variable", "valor"]]

# --- ORDENAR ---
df_final = df_final.sort_values(by=id_vars + ["hour"]).reset_index(drop=True)

# --- GUARDAR ---
df_final.to_csv("insolacion_1.csv", index=False, encoding="utf-8-sig")
print("[OK] Archivo transformado guardado como 'insolacion_1.csv'")
print(df_final.head(10))
