import pandas as pd

INPUT_CSV = "insolacion_1.csv"
OUTPUT_CSV = "insolacion_1_completas_15min_1.csv"

# --- CARGAR Y LIMPIAR ---
df = pd.read_csv(INPUT_CSV)
df.columns = df.columns.str.strip().str.replace('\ufeff', '')  # limpiar espacios y BOM

# --- COLUMNAS DE VALORES ---
value_cols = [c for c in df.columns if c not in ['YEAR', 'MONTH', 'DAY', 'hour']]

if not value_cols:
    raise ValueError("No se detectaron columnas de valores a expandir.")

all_hours = list(range(24))
expanded_rows = []

# --- RELLENAR HORAS FALTANTES ---
for (y, m, d), group in df.groupby(["YEAR", "MONTH", "DAY"]):
    hour_dicts = {col: dict(zip(group["hour"], group[col])) for col in value_cols}

    for h in all_hours:
        new_row = {"YEAR": y, "MONTH": m, "DAY": d, "hour": h}
        for col in value_cols:
            prev_hours = [hh for hh in sorted(hour_dicts[col].keys()) if hh <= h]
            if prev_hours:
                last_hour = prev_hours[-1]
                new_row[col] = hour_dicts[col][last_hour]
            else:
                new_row[col] = None
        expanded_rows.append(new_row)

# --- CREAR Y GUARDAR ---
df_expanded = pd.DataFrame(expanded_rows)
df_expanded = df_expanded.sort_values(by=["YEAR", "MONTH", "DAY", "hour"]).reset_index(drop=True)

df_expanded.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

print(f"[FIN] Archivo generado: {OUTPUT_CSV}")
print(df_expanded.head(24))
