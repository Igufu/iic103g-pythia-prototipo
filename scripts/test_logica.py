"""Tests unitarios de la logica pura del pipeline (sin modelo ni dataset).
Corre rapido y sin GPU. Si falla => bug en el indexado o en las metricas.
"""
from __future__ import annotations
import math
import numpy as np
import torch
from scipy import stats as scipy_stats


def test_topk_indexado_excluye_bos():
    """La reconstruccion (doc_idx, pos_idx) desde topk debe (a) no incluir pos 0
    y (b) devolver exactamente el valor original en acts[doc, pos, neurona]."""
    torch.manual_seed(0)
    B, T, N = 4, 9, 3  # 9 = BOS + 8 posiciones validas
    acts = torch.randn(B, T, N)
    acts_valid = acts[:, 1:, :]  # (B, 8, N)
    P_VALID = acts_valid.shape[1]
    flat = acts_valid.reshape(B * P_VALID, N)
    vals, idx = flat.topk(5, dim=0)
    doc_idx = (idx // P_VALID).numpy()
    pos_idx = (idx % P_VALID + 1).numpy()
    vals_np = vals.numpy()
    assert (pos_idx >= 1).all(), "posiciones deberian ser >= 1"
    assert (pos_idx <= T - 1).all(), "posiciones fuera de rango"
    # el valor reconstruido coincide con el top-k reportado
    for rank in range(5):
        for n in range(N):
            d = int(doc_idx[rank, n])
            p = int(pos_idx[rank, n])
            assert math.isclose(float(acts[d, p, n]), float(vals_np[rank, n]), abs_tol=1e-6), \
                f"desalineado en rank={rank} n={n}: {acts[d,p,n]} vs {vals_np[rank,n]}"


def test_topk_son_los_maximos_reales():
    """El top-k por neurona debe coincidir con sort descendente de la columna
    (excluyendo pos 0)."""
    torch.manual_seed(1)
    B, T, N = 3, 7, 4
    acts = torch.randn(B, T, N)
    valid = acts[:, 1:, :].reshape(-1, N)
    vals, _ = valid.topk(3, dim=0)
    for n in range(N):
        sorted_desc, _ = valid[:, n].sort(descending=True)
        assert torch.allclose(vals[:, n], sorted_desc[:3])


def test_entropia_shannon_casos_borde():
    """Caso 1 fuente -> 0 bits. k fuentes equiprobables -> log2(k)."""
    def shannon(labels):
        _, c = np.unique(labels, return_counts=True)
        p = c / c.sum()
        return float(-np.sum(p * np.log2(p)))
    assert math.isclose(shannon(["a"] * 10), 0.0, abs_tol=1e-9)
    assert math.isclose(shannon(["a", "b"]), 1.0, abs_tol=1e-9)
    assert math.isclose(shannon(["a", "b", "c", "d"]), 2.0, abs_tol=1e-9)
    # mezcla no uniforme en el rango esperado
    e = shannon(["a"] * 8 + ["b"] * 2)
    assert 0.0 < e < 1.0


def test_kurtosis_normal_cerca_de_cero():
    """Para muestras normales grandes, kurtosis (Fisher) ~ 0."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal((50_000, 4))
    k = scipy_stats.kurtosis(x, axis=0, fisher=True)
    assert (np.abs(k) < 0.1).all(), f"kurtosis fuera de rango esperado: {k}"


def test_posiciones_son_1_a_128_despues_de_bos():
    """Simula la regla de +1 que usa 02_extraer.py y confirma que
    cubre toda la ventana no-BOS."""
    P_VALID = 128
    simulated_valid = np.arange(P_VALID)
    posiciones = simulated_valid + 1
    assert posiciones.min() == 1
    assert posiciones.max() == 128


TESTS = [
    test_topk_indexado_excluye_bos,
    test_topk_son_los_maximos_reales,
    test_entropia_shannon_casos_borde,
    test_kurtosis_normal_cerca_de_cero,
    test_posiciones_son_1_a_128_despues_de_bos,
]

if __name__ == "__main__":
    import sys
    fails = []
    for t in TESTS:
        try:
            t()
            print(f"  [OK]   {t.__name__}")
        except Exception as e:
            fails.append((t.__name__, e))
            print(f"  [FAIL] {t.__name__}: {e}")
    if fails:
        print(f"\n{len(fails)}/{len(TESTS)} tests fallaron")
        sys.exit(1)
    print(f"\n{len(TESTS)}/{len(TESTS)} tests OK")
