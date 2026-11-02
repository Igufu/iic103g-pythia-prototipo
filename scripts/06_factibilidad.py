"""Lee stats.json y tests.json y escribe FACTIBILIDAD.md con las cifras medidas.
Ningun numero esta hardcodeado: todo sale de archivos producidos por 01-05."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATOS = ROOT / "datos"

with open(DATOS / "stats.json") as f:
    stats = json.load(f)
with open(DATOS / "tests.json") as f:
    tests = json.load(f)
with open(DATOS / "dataset_meta.json") as f:
    dmeta = json.load(f)

act = pd.read_csv(DATOS / "activaciones.csv")

rows = stats["por_checkpoint"]
n_ckpt = len(rows)
csv_mb_total = stats["csv_mb_total"]
filas_total = stats["filas_total"]
device = stats["device"]

lines: list[str] = []
lines.append("# FACTIBILIDAD — Prototipo Pythia-70m (IIC103G)")
lines.append("")
lines.append(f"- Dispositivo: `{device}`  dtype: `{stats['dtype']}`")
lines.append(f"- Docs: {dmeta['target_docs']} de pile-10k (>= {dmeta['seq_len']} tokens, {dmeta['descartados_por_longitud']} descartados)")
lines.append(f"- Fuentes pile unicas en el subset: {dmeta['n_fuentes']}")
lines.append(f"- Checkpoints procesados: {[r['checkpoint'] for r in rows]}")
lines.append(f"- Capas detectadas: {stats['n_layers_detectado']}  neuronas MLP: {stats['d_mlp_detectado']}")
lines.append("")

lines.append("## SUPUESTOS")
lines.append("")
lines.append("- `fuente_pile` = `meta.pile_set_name` del dataset NeelNanda/pile-10k.")
lines.append("- Cada secuencia lleva BOS prependido => 129 tokens; posicion 0 = BOS y se excluye del top-k.")
lines.append("- Kurtosis se calcula sobre los 200x129 = 25800 tokens (incluye BOS) porque el encargo pide 'todos los tokens'.")
lines.append("- Contexto izq/der: 5 tokens decodificados a cada lado.")
lines.append("- fp16 cuando hay GPU, fp32 en CPU.")
lines.append("- Un unico CSV `activaciones.csv` para los 3 checkpoints.")
lines.append("")

# ---- tabla principal
lines.append("## Tiempos y tamanio por checkpoint")
lines.append("")
lines.append("| checkpoint | segundos | MB CSV (prorrateado) | filas | RSS (MB) |")
lines.append("|---:|---:|---:|---:|---:|")
tot_s = 0.0
for r in rows:
    tot_s += r["segundos"]
    lines.append(
        f"| {r['checkpoint']} | {r['segundos']:.1f} | {r['mb_csv_prorrateado']:.2f} | {r['filas_producidas']} | {r['rss_despues_mb']:.0f} |"
    )
lines.append(f"| **total** | **{tot_s:.1f}** | **{csv_mb_total:.2f}** | **{filas_total}** |  |")
lines.append("")

# ---- proyeccion a 20 checkpoints
avg_s = tot_s / max(1, n_ckpt)
filas_por_ckpt = filas_total / max(1, n_ckpt)
mb_por_ckpt = csv_mb_total / max(1, n_ckpt)

p20_k20_s = avg_s * 20
p20_k20_mb = mb_por_ckpt * 20
p20_k20_f = filas_por_ckpt * 20

# k=50: tiempo del forward casi no cambia; top-k y escritura de CSV crecen ~2.5x.
# Conservador: tiempo = avg_s * 20 * 1.1; CSV y filas * 2.5
p20_k50_s = avg_s * 20 * 1.1
p20_k50_mb = mb_por_ckpt * 20 * (50 / 20)
p20_k50_f = filas_por_ckpt * 20 * (50 / 20)

lines.append("## Proyeccion a 20 checkpoints")
lines.append("")
lines.append("Asume que el tiempo escala lineal en #checkpoints (el forward domina) y que ")
lines.append("CSV/filas escalan lineal en k. Para k=50 se agrega 10% extra de tiempo por el top-k y el I/O.")
lines.append("")
lines.append("| escenario | tiempo (s) | tiempo (min) | tamanio CSV (MB) | filas |")
lines.append("|---|---:|---:|---:|---:|")
lines.append(f"| 20 ckpt x k=20 | {p20_k20_s:.0f} | {p20_k20_s/60:.1f} | {p20_k20_mb:.1f} | {int(p20_k20_f)} |")
lines.append(f"| 20 ckpt x k=50 | {p20_k50_s:.0f} | {p20_k50_s/60:.1f} | {p20_k50_mb:.1f} | {int(p20_k50_f)} |")
lines.append("")

# ---- tests
lines.append("## Resultados estadisticos (PRELIMINARES)")
lines.append("")
lines.append(f"Comparacion: {tests.get('comparacion','?')}  n = {tests['n']} neuronas pareadas.")
w = tests.get("wilcoxon", {})
if "error" in w:
    lines.append(f"- Wilcoxon pareado: ERROR ({w['error']})")
else:
    lines.append(f"- Wilcoxon pareado: stat = {w['stat']:.4g}  p = {w['p']:.4g}")
s = tests.get("spearman", {})
if "error" in s:
    lines.append(f"- Spearman: ERROR ({s['error']})")
else:
    lines.append(f"- Spearman: rho = {s['rho']:.4g}  p = {s['p']:.4g}")
lines.append("")
lines.append("Son numeros de prototipo (3 checkpoints, 200 docs, top-20). No se interpreta la hipotesis aqui.")
lines.append("")

# ---- problemas
lines.append("## Problemas encontrados")
lines.append("")
probs = stats.get("problemas", [])
if probs:
    for p in probs:
        lines.append(f"- {p}")
else:
    lines.append("- Ninguno registrado automaticamente durante la ejecucion.")
lines.append("")

# ---- recomendacion automatica basada en el peor escenario proyectado
lines.append("## Recomendacion")
lines.append("")
if p20_k50_s < 1800 and p20_k50_mb < 1024:
    rec = ("Factible escalar tal cual a 20 checkpoints con k=50 (menos de 30 min y menos de 1 GB de CSV). "
           "Si el host de produccion es Colab con GPU, el margen es amplio.")
elif p20_k20_s < 3600 and p20_k20_mb < 1024:
    rec = ("Factible con k=20 (menos de 1 h). k=50 puede ser pesado: recomiendo dejar k=20 "
           "o reducir a 10-12 checkpoints en escala logaritmica.")
else:
    rec = ("No factible tal cual. Opciones: (a) correr en Colab GPU, (b) reducir docs a 100 o seq_len a 64, "
           "(c) procesar en streaming sin cachear todas las capas a la vez.")
lines.append(f"- {rec}")
lines.append("")

(ROOT / "FACTIBILIDAD.md").write_text("\n".join(lines))
print(f"[06] Escrito {ROOT/'FACTIBILIDAD.md'}")
