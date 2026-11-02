"""Calcula entropia Shannon de fuente_pile en el top-k (por neurona y checkpoint)
y la une con la kurtosis guardada en 02_extraer.py."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATOS = ROOT / "datos"

act = pd.read_csv(DATOS / "activaciones.csv")
kurt = pd.read_csv(DATOS / "kurtosis.csv")


def shannon(labels: pd.Series) -> float:
    _, counts = np.unique(labels.to_numpy(), return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p)))


ent_series = act.groupby(["step", "capa", "neurona"], sort=False)["fuente_pile"].apply(shannon)
ent_series.name = "entropia_fuente"
ent = ent_series.reset_index()

metricas = ent.merge(kurt, on=["step", "capa", "neurona"], how="left")
metricas.to_csv(DATOS / "metricas.csv", index=False)
print(f"[03] Metricas: {len(metricas)} filas en {DATOS/'metricas.csv'}")
print(f"[03] entropia_fuente  min={metricas['entropia_fuente'].min():.3f}  max={metricas['entropia_fuente'].max():.3f}")
print(f"[03] kurtosis  min={metricas['kurtosis'].min():.2f}  max={metricas['kurtosis'].max():.2f}")
