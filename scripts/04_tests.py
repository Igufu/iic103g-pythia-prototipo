"""Wilcoxon pareado y Spearman de la entropia entre step 1000 y 143000."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from scipy.stats import wilcoxon, spearmanr

ROOT = Path(__file__).resolve().parent.parent
DATOS = ROOT / "datos"

m = pd.read_csv(DATOS / "metricas.csv")

a = m.loc[m["step"] == 1000, ["capa", "neurona", "entropia_fuente"]].rename(
    columns={"entropia_fuente": "e_a"}
)
b = m.loc[m["step"] == 143000, ["capa", "neurona", "entropia_fuente"]].rename(
    columns={"entropia_fuente": "e_b"}
)
par = a.merge(b, on=["capa", "neurona"])
n = len(par)

resultado: dict = {"n": int(n), "comparacion": "step 1000 vs step 143000"}

def _unpack(obj, attr_names):
    for a in attr_names:
        v = getattr(obj, a, None)
        if v is not None:
            return v
    try:
        return obj[0]  # type: ignore[index]
    except Exception:
        return None


try:
    w = wilcoxon(par["e_a"].to_numpy(), par["e_b"].to_numpy())
    w_stat = _unpack(w, ["statistic"])
    w_p = _unpack(w, ["pvalue"])
    resultado["wilcoxon"] = {"stat": float(w_stat), "p": float(w_p)}  # type: ignore[arg-type]
except Exception as e:
    resultado["wilcoxon"] = {"error": str(e)}

try:
    s = spearmanr(par["e_a"].to_numpy(), par["e_b"].to_numpy())
    s_rho = _unpack(s, ["statistic", "correlation"])
    s_p = _unpack(s, ["pvalue"])
    resultado["spearman"] = {"rho": float(s_rho), "p": float(s_p)}  # type: ignore[arg-type]
except Exception as e:
    resultado["spearman"] = {"error": str(e)}

with open(DATOS / "tests.json", "w") as f:
    json.dump(resultado, f, indent=2, ensure_ascii=False)
print(f"[04] n={n}")
print(json.dumps(resultado, indent=2, ensure_ascii=False))
