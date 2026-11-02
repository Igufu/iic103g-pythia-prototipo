"""Chain of verification: cinco chequeos que la salida satisface las condiciones del encargo.
Imprime OK/FAIL por chequeo; si alguno falla, levanta SystemExit(1)."""
from __future__ import annotations
import json
import math
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATOS = ROOT / "datos"

act = pd.read_csv(DATOS / "activaciones.csv")
m = pd.read_csv(DATOS / "metricas.csv")
with open(DATOS / "stats.json") as f:
    stats = json.load(f)

n_ckpt = len(stats["por_checkpoint"])
n_layers = stats["n_layers_detectado"] or 6
d_mlp = stats["d_mlp_detectado"] or 2048

chequeos = []

# 1. filas CSV == n_ckpt * n_layers * d_mlp * 20
esperado_filas = n_ckpt * n_layers * d_mlp * 20
obtenido_filas = len(act)
chequeos.append(("filas CSV == n_ckpt * n_layers * d_mlp * 20",
                 esperado_filas, obtenido_filas, esperado_filas == obtenido_filas))

# 2. 12 columnas exactas, en el orden pedido
cols_esperadas = [
    "step", "capa", "neurona", "rank", "activacion",
    "token_id", "token_texto", "doc_id", "posicion",
    "contexto_izq", "contexto_der", "fuente_pile",
]
chequeos.append(("columnas en orden esperado",
                 cols_esperadas, list(act.columns),
                 list(act.columns) == cols_esperadas))

# 3. cada (step, capa, neurona) tiene exactamente 20 filas (ranks 0..19)
grp = act.groupby(["step", "capa", "neurona"]).size()
ok3 = bool((grp == 20).all())
chequeos.append(("cada (step,capa,neurona) tiene 20 filas",
                 True, ok3, ok3))

# 4. no hay posicion == 0 (BOS excluido)
ok4 = bool((act["posicion"] != 0).all())
chequeos.append(("ninguna fila con posicion==0", True, ok4, ok4))

# 5. entropia esta en [0, log2(n_fuentes)]
n_fuentes_global = act["fuente_pile"].nunique()
cota = math.log2(n_fuentes_global) if n_fuentes_global > 1 else 0.0
ok5 = bool(((m["entropia_fuente"] >= -1e-9) & (m["entropia_fuente"] <= cota + 1e-6)).all())
chequeos.append((f"entropia en [0, log2({n_fuentes_global})={cota:.3f}]",
                 True, ok5, ok5))

print("Chain of verification:")
all_ok = True
for i, (nombre, esperado, obtenido, ok) in enumerate(chequeos, 1):
    tag = "OK  " if ok else "FAIL"
    print(f"  [{tag}] {i}. {nombre}")
    print(f"         esperado: {esperado}")
    print(f"         obtenido: {obtenido}")
    all_ok = all_ok and ok

if not all_ok:
    raise SystemExit(1)
print("Todos los chequeos pasaron.")
