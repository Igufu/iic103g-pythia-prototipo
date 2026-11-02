"""Extrae top-k=20 activaciones de blocks.{l}.mlp.hook_post para cada neurona,
en los checkpoints {0, 1000, 143000} de Pythia-70m. Mide tiempo y memoria.
Tambien calcula kurtosis por neurona sobre todos los tokens (incluyendo BOS).

Decisiones (SUPUESTOs, registrados en FACTIBILIDAD.md):
- Se prepone BOS a cada doc, asi la secuencia es 129 tokens; posicion 0 = BOS.
- Top-k excluye posicion 0.
- Kurtosis se calcula sobre los 200*129 = 25800 tokens (incluye BOS) porque el
  encargo pide 'todos los tokens' sin ser explicito.
- Contexto: 5 tokens a cada lado (en texto decodificado).
- fp16 si hay GPU, fp32 si CPU.
"""
from __future__ import annotations
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
from scipy import stats as scipy_stats
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
DATOS = ROOT / "datos"
DATOS.mkdir(parents=True, exist_ok=True)

CHECKPOINTS = [0, 1000, 143000]
TOP_K = 20
CTX_WINDOW = 5

problemas: list[str] = []

df_docs = pd.read_parquet(DATOS / "dataset.parquet")
N_DOCS = len(df_docs)
assert N_DOCS == 200, f"Esperaba 200 docs, hay {N_DOCS}"
SEQ_LEN = len(df_docs.iloc[0]["tokens"])  # 128
assert SEQ_LEN == 128

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
print(f"[02] device={device} dtype={dtype}")

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
if bos_id is None:
    bos_id = 0
    problemas.append("Tokenizer sin bos/eos; se uso 0 como BOS.")

tokens_list = [[bos_id] + list(row["tokens"]) for _, row in df_docs.iterrows()]
tokens_np = np.asarray(tokens_list, dtype=np.int64)  # (200, 129)
tokens_tensor = torch.tensor(tokens_np, device=device)
SEQ_PLUS = tokens_tensor.shape[1]  # 129
P_VALID = SEQ_PLUS - 1  # 128 posiciones no-BOS

# Decodificar cada token individualmente para contextos
decoded_per_doc = [[tokenizer.decode([int(t)]) for t in row] for row in tokens_np]
fuentes_arr = df_docs["fuente_pile"].to_numpy()

stats_rows = []
act_frames = []
kurt_frames = []
n_layers_detectado: int | None = None
d_mlp_last: int | None = None

for cp in CHECKPOINTS:
    print(f"\n[02] === checkpoint {cp} ===")
    proc = psutil.Process(os.getpid())
    rss0 = proc.memory_info().rss / 1e6
    t0 = time.time()

    try:
        model = HookedTransformer.from_pretrained("pythia-70m", checkpoint_value=cp)
    except Exception as e:
        problemas.append(f"No pude cargar checkpoint {cp}: {e}")
        print(f"[02] ERROR cargando ckpt {cp}: {e}")
        continue
    model = model.to(device)
    if dtype == torch.float16:
        model = model.half()
    model.eval()

    n_layers = int(model.cfg.n_layers)
    d_mlp = int(model.cfg.d_mlp)
    n_layers_detectado = n_layers
    d_mlp_last = d_mlp

    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens_tensor,
            names_filter=lambda n: n.endswith("mlp.hook_post"),
            return_type=None,
        )

    cp_rows_count = 0
    for layer in range(n_layers):
        key = f"blocks.{layer}.mlp.hook_post"
        acts = cache[key].float()  # (200, 129, d_mlp)

        # ---- kurtosis sobre todos los tokens (incluye BOS)
        acts_all = acts.reshape(-1, d_mlp).cpu().numpy()
        kurts = scipy_stats.kurtosis(acts_all, axis=0, fisher=True)
        kurt_frames.append(pd.DataFrame({
            "step": cp,
            "capa": layer,
            "neurona": np.arange(d_mlp, dtype=np.int32),
            "kurtosis": kurts.astype(np.float32),
        }))

        # ---- top-k excluyendo posicion 0 (BOS)
        acts_valid = acts[:, 1:, :]  # (200, 128, d_mlp)
        acts2d = acts_valid.reshape(N_DOCS * P_VALID, d_mlp)
        topk_vals, topk_idx = acts2d.topk(TOP_K, dim=0)  # (20, d_mlp)
        topk_vals = topk_vals.cpu().numpy()
        topk_idx = topk_idx.cpu().numpy()

        ranks = np.tile(np.arange(TOP_K, dtype=np.int32).reshape(-1, 1), (1, d_mlp))
        neurons = np.tile(np.arange(d_mlp, dtype=np.int32).reshape(1, -1), (TOP_K, 1))
        doc_idx = (topk_idx // P_VALID).astype(np.int32)
        pos_idx = (topk_idx % P_VALID + 1).astype(np.int32)  # +1 para re-indexar sobre 0..128

        d_flat = doc_idx.flatten()
        p_flat = pos_idx.flatten()

        token_ids = tokens_np[d_flat, p_flat].astype(np.int32)
        token_texts = [decoded_per_doc[d][p] for d, p in zip(d_flat, p_flat)]
        ctx_izq, ctx_der = [], []
        for d, p in zip(d_flat, p_flat):
            lo = max(0, p - CTX_WINDOW)
            hi = min(SEQ_PLUS, p + CTX_WINDOW + 1)
            ctx_izq.append("".join(decoded_per_doc[d][lo:p]))
            ctx_der.append("".join(decoded_per_doc[d][p + 1:hi]))
        fuentes = fuentes_arr[d_flat]

        layer_df = pd.DataFrame({
            "step": np.int32(cp),
            "capa": np.int32(layer),
            "neurona": neurons.flatten(),
            "rank": ranks.flatten(),
            "activacion": topk_vals.flatten().astype(np.float32),
            "token_id": token_ids,
            "token_texto": token_texts,
            "doc_id": d_flat,
            "posicion": p_flat,
            "contexto_izq": ctx_izq,
            "contexto_der": ctx_der,
            "fuente_pile": fuentes,
        })
        act_frames.append(layer_df)
        cp_rows_count += len(layer_df)

        del acts, acts_valid, acts2d, acts_all, topk_vals, topk_idx
        gc.collect()

    del cache, model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    elapsed = time.time() - t0
    rss1 = proc.memory_info().rss / 1e6
    stats_rows.append({
        "checkpoint": int(cp),
        "segundos": round(elapsed, 2),
        "rss_despues_mb": round(rss1, 1),
        "delta_rss_mb": round(rss1 - rss0, 1),
        "filas_producidas": int(cp_rows_count),
    })
    print(f"[02] ckpt={cp}  tiempo={elapsed:.1f}s  RSS={rss1:.0f}MB  filas={cp_rows_count}")

# -------- guardar
act_df = pd.concat(act_frames, ignore_index=True)
act_df = act_df[[
    "step", "capa", "neurona", "rank", "activacion",
    "token_id", "token_texto", "doc_id", "posicion",
    "contexto_izq", "contexto_der", "fuente_pile",
]]
csv_path = DATOS / "activaciones.csv"
act_df.to_csv(csv_path, index=False)
csv_mb = csv_path.stat().st_size / 1e6

kurt_df = pd.concat(kurt_frames, ignore_index=True)
kurt_df.to_csv(DATOS / "kurtosis.csv", index=False)

# bytes CSV por checkpoint ~ proporcional a filas
total_filas = len(act_df)
for s in stats_rows:
    s["mb_csv_prorrateado"] = round(csv_mb * s["filas_producidas"] / max(1, total_filas), 2)

info = {
    "device": device,
    "dtype": str(dtype),
    "csv_mb_total": round(csv_mb, 2),
    "filas_total": int(total_filas),
    "por_checkpoint": stats_rows,
    "problemas": problemas,
    "checkpoints": CHECKPOINTS,
    "top_k": TOP_K,
    "seq_len": SEQ_LEN,
    "n_docs": N_DOCS,
    "n_layers_detectado": n_layers_detectado,
    "d_mlp_detectado": d_mlp_last,
}
with open(DATOS / "stats.json", "w") as f:
    json.dump(info, f, indent=2, ensure_ascii=False)

print(f"\n[02] CSV: {csv_path}  ({csv_mb:.1f} MB, {total_filas} filas)")
print(f"[02] stats: {DATOS/'stats.json'}")
if problemas:
    print(f"[02] Problemas: {problemas}")
