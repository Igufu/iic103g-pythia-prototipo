"""Smoke test del entorno. Corre antes del pipeline principal para fallar
rapido con mensajes claros si algo del entorno esta mal.

Chequea:
 1. imports (torch, transformer_lens, datasets, scipy, matplotlib);
 2. API de TL: HookedTransformer.from_pretrained con checkpoint_value;
 3. forward con cache de blocks.{l}.mlp.hook_post en un batch pequenio;
 4. dtypes consistentes (nada en fp16 por accidente);
 5. top-k por neurona y kurtosis sobre un tensor dummy producen formas correctas;
 6. acceso a datasets (NeelNanda/pile-10k) con una muestra de 1 fila.
"""
from __future__ import annotations
import sys
import traceback

FAIL = []

def ok(msg: str) -> None:
    print(f"  [OK]   {msg}")

def fail(msg: str, err: BaseException | None = None) -> None:
    FAIL.append(msg)
    print(f"  [FAIL] {msg}")
    if err is not None:
        print("         " + "".join(traceback.format_exception_only(type(err), err)).strip())

print("\n== 1. imports ==")
try:
    import torch
    import numpy as np
    import pandas as pd
    import scipy
    import matplotlib
    import datasets as hf_datasets
    import transformers
    import transformer_lens
    ok(f"torch {torch.__version__}  numpy {np.__version__}  pandas {pd.__version__}")
    ok(f"scipy {scipy.__version__}  matplotlib {matplotlib.__version__}")
    ok(f"transformers {transformers.__version__}  datasets {hf_datasets.__version__}")
    ok(f"transformer_lens {transformer_lens.__version__}")
except Exception as e:
    fail("imports basicos", e)

print("\n== 2. API TransformerLens: HookedTransformer existe ==")
try:
    from transformer_lens import HookedTransformer  # noqa: F401
    ok("HookedTransformer importable (TL < 4.0)")
except Exception as e:
    fail("HookedTransformer no importable; probablemente TL >= 4.0", e)

print("\n== 3. cargar pythia-70m checkpoint 1000 + forward con cache ==")
model = None
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"  # type: ignore[name-defined]
    model = HookedTransformer.from_pretrained("pythia-70m", checkpoint_value=1000)  # type: ignore[name-defined]
    model = model.to(device).eval()
    ok(f"modelo cargado en {device}; n_layers={model.cfg.n_layers} d_mlp={model.cfg.d_mlp}")
except Exception as e:
    fail("carga de pythia-70m@1000", e)

if model is not None:
    try:
        toy = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], device=device)  # type: ignore[name-defined]
        with torch.no_grad():  # type: ignore[name-defined]
            _, cache = model.run_with_cache(
                toy,
                names_filter=lambda n: n.endswith("mlp.hook_post"),
                return_type=None,
            )
        hooks = [k for k in cache.keys() if k.endswith("mlp.hook_post")]
        if len(hooks) == model.cfg.n_layers:
            ok(f"run_with_cache: {len(hooks)} hooks mlp.hook_post")
        else:
            fail(f"se esperaban {model.cfg.n_layers} hooks, hay {len(hooks)}")
        sample = cache[hooks[0]]
        if sample.dtype != torch.float32:  # type: ignore[name-defined]
            fail(f"activaciones no son fp32: {sample.dtype}")
        else:
            ok(f"activaciones en fp32, shape={tuple(sample.shape)}")
    except Exception as e:
        fail("forward con cache", e)

print("\n== 4. top-k por neurona + kurtosis sobre tensor dummy ==")
try:
    import torch as _t
    from scipy import stats as _st
    rng = _t.Generator().manual_seed(0)
    x = _t.randn(2, 7, 16, generator=rng)  # (B, T, N)
    xv = x[:, 1:, :]  # excluir pos 0
    flat = xv.reshape(-1, 16)
    vals, idx = flat.topk(5, dim=0)
    assert vals.shape == (5, 16), f"topk shape {vals.shape}"
    k = _st.kurtosis(x.reshape(-1, 16).numpy(), axis=0, fisher=True)
    assert k.shape == (16,), f"kurtosis shape {k.shape}"
    ok(f"top-k shape {tuple(vals.shape)} y kurtosis shape {k.shape}")
except Exception as e:
    fail("top-k / kurtosis sobre dummy", e)

print("\n== 5. acceso a NeelNanda/pile-10k (streaming, 1 muestra) ==")
try:
    import datasets as hf_datasets  # noqa: F811
    ds = hf_datasets.load_dataset("NeelNanda/pile-10k", split="train", streaming=True)
    first = next(iter(ds))
    assert "text" in first and "meta" in first and "pile_set_name" in first["meta"]
    ok(f"pile-10k accesible; primera fuente='{first['meta']['pile_set_name']}'")
except Exception as e:
    fail("carga de pile-10k", e)

print("\n== resumen ==")
if FAIL:
    for m in FAIL:
        print(f"  FAIL: {m}")
    sys.exit(1)
print("  Entorno OK, se puede correr el pipeline.")
