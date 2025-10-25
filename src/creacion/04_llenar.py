import pandas as pd

INPUT_CSV = "insolacion_1_completas_15min.csv"
OUTPUT_CSV = "insolacion_1_completas_15min_1.csv"

# Cargar CSV y limpiar nombres
df = pd.read_csv(INPUT_CSV)
df.columns = df.columns.str.strip().str.replace('\ufeff','')

# Columnas de datos (todo lo que no sea year, month, day, hour, minute)
data_cols = [c for c in df.columns if c not in ["year","month","day","hour","minute"]]

# Crear columna fecha para manejo de días
df["fecha"] = pd.to_datetime(df[["year","month","day"]])

# Ordenar por fecha, hora y minuto
df = df.sort_values(["fecha","hour","minute"]).reset_index(drop=True)

# Crear rango completo de fechas (todos los días)
fecha_completa = pd.date_range(df["fecha"].min(), df["fecha"].max(), freq="D")

relleno = []
ultimo_dia = None

for fecha in fecha_completa:
    filas_dia = df[df["fecha"] == fecha]
    if filas_dia.empty:
        if ultimo_dia is not None:
            # Copiar último día existente
            filas_dia = ultimo_dia.copy()
            filas_dia["fecha"] = fecha
            filas_dia["year"] = fecha.year
            filas_dia["month"] = fecha.month
            filas_dia["day"] = fecha.day
    else:
        ultimo_dia = filas_dia
    relleno.append(filas_dia)

# Concatenar todos los días
df_final = pd.concat(relleno, ignore_index=True)

# Orden final y limpiar columna auxiliar
df_final = df_final[["year","month","day","hour","minute"] + data_cols]
df_final = df_final.sort_values(["year","month","day","hour","minute"]).reset_index(drop=True)

# Guardar CSV
df_final.to_csv(OUTPUT_CSV, index=False)
print(f"[FIN] Archivo generado: {OUTPUT_CSV}")
