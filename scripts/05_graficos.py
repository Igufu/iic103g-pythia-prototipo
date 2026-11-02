"""Dos graficos: lineas (entropia media por step y capa) y dispersion (early vs final)."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATOS = ROOT / "datos"
GRAF = ROOT / "graficos"
GRAF.mkdir(parents=True, exist_ok=True)

m = pd.read_csv(DATOS / "metricas.csv")

# --- lineas: entropia media por step y capa
mean_e = m.groupby(["step", "capa"])["entropia_fuente"].mean().reset_index()
fig, ax = plt.subplots(figsize=(8, 5))
for layer, g in mean_e.groupby("capa"):
    g = g.sort_values("step")
    ax.plot(g["step"], g["entropia_fuente"], marker="o", label=f"capa {layer}")
ax.set_xscale("symlog")
ax.set_xlabel("step (symlog)")
ax.set_ylabel("entropia media de fuente_pile en top-k")
ax.set_title("Entropia media por step y capa (prototipo, 3 checkpoints)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(GRAF / "lineas_entropia.png", dpi=120)
plt.close(fig)

# --- dispersion: entropia step 1000 vs step 143000
a = m.loc[m["step"] == 1000, ["capa", "neurona", "entropia_fuente"]].rename(columns={"entropia_fuente": "e1000"})
b = m.loc[m["step"] == 143000, ["capa", "neurona", "entropia_fuente"]].rename(columns={"entropia_fuente": "e143000"})
par = a.merge(b, on=["capa", "neurona"])

fig, ax = plt.subplots(figsize=(6, 6))
sc = ax.scatter(par["e1000"], par["e143000"], s=3, alpha=0.4, c=par["capa"], cmap="viridis")
hi = max(par["e1000"].max(), par["e143000"].max())
ax.plot([0, hi], [0, hi], "k--", alpha=0.4)
ax.set_xlabel("entropia step 1000")
ax.set_ylabel("entropia step 143000")
ax.set_title("Entropia por neurona: temprana vs final")
cb = fig.colorbar(sc, ax=ax, label="capa")
fig.tight_layout()
fig.savefig(GRAF / "scatter_entropia.png", dpi=120)
plt.close(fig)

print(f"[05] Guardados {GRAF/'lineas_entropia.png'} y {GRAF/'scatter_entropia.png'}")
