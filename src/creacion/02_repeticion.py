import pandas as pd

# Configuración
INPUT_CSV = "insolacion_1_completas.csv"
OUTPUT_CSV = "insolacion_1_completas_15min.csv"

# Cargar CSV
df = pd.read_csv(INPUT_CSV)

# Crear lista de minutos
minutes = [0, 15, 30, 45]

# Lista para almacenar filas expandidas
expanded_rows = []

# Repetir cada fila 4 veces con los minutos adecuados
for _, row in df.iterrows():
    for m in minutes:
        new_row = row.copy()
        new_row["minute"] = m
        expanded_rows.append(new_row)

# Crear dataframe final
df_expanded = pd.DataFrame(expanded_rows)

# Ordenar por fecha y hora
df_expanded = df_expanded.sort_values(by=["YEAR", "MONTH", "DAY", "hour", "minute"]).reset_index(drop=True)

# Guardar CSV
df_expanded.to_csv(OUTPUT_CSV, index=False)
print(f"[FIN] Archivo generado: {OUTPUT_CSV}")
print(df_expanded.head(16))
