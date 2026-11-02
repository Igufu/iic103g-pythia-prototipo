"""Carga NeelNanda/pile-10k y guarda 200 docs con >=128 tokens, truncados a 128.

Columnas de salida (parquet): doc_id, pile_idx, fuente_pile, tokens, texto.
`fuente_pile` sale de `meta.pile_set_name` (SUPUESTO documentado en FACTIBILIDAD.md).
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
DATOS = ROOT / "datos"
DATOS.mkdir(parents=True, exist_ok=True)

TARGET_DOCS = 200
SEQ_LEN = 128

ds = load_dataset("NeelNanda/pile-10k", split="train")
tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")

rows = []
descartados = 0
for i, ex in enumerate(ds):
    ids = tok.encode(ex["text"], add_special_tokens=False)
    if len(ids) < SEQ_LEN:
        descartados += 1
        continue
    rows.append({
        "doc_id": len(rows),
        "pile_idx": i,
        "fuente_pile": ex["meta"]["pile_set_name"],
        "tokens": ids[:SEQ_LEN],
        "texto": ex["text"][:3000],
    })
    if len(rows) >= TARGET_DOCS:
        break

if len(rows) < TARGET_DOCS:
    raise RuntimeError(
        f"Solo {len(rows)}/{TARGET_DOCS} docs con >= {SEQ_LEN} tokens. "
        f"Descartados: {descartados}."
    )

df = pd.DataFrame(rows)
df.to_parquet(DATOS / "dataset.parquet")

# metadatos pequeños para script 06
meta = {
    "target_docs": TARGET_DOCS,
    "seq_len": SEQ_LEN,
    "descartados_por_longitud": descartados,
    "fuentes_pile_unicas": sorted(df["fuente_pile"].unique().tolist()),
    "n_fuentes": int(df["fuente_pile"].nunique()),
}
with open(DATOS / "dataset_meta.json", "w") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

print(f"[01] Guardados {len(df)} docs en {DATOS/'dataset.parquet'}")
print(f"[01] {meta['n_fuentes']} fuentes distintas en el subset")
