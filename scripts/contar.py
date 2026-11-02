"""Imprime filas x columnas del CSV de activaciones."""
from pathlib import Path
import pandas as pd

CSV = Path(__file__).resolve().parent.parent / "datos" / "activaciones.csv"
df = pd.read_csv(CSV)
print(f"{df.shape[0]} filas x {df.shape[1]} columnas")
print(f"columnas: {list(df.columns)}")
