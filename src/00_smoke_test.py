"""
00_smoke_test.py — Tests ALL the code on small SYNTHETIC data, without touching the real data.

Run it first on the cluster: if it ends with "ALL TESTS PASSED", the environment and the code work.
It runs on CPU in a couple of minutes and writes only to a temporary directory.

    srun -p long -c 4 --mem=16G -t 00:30:00 python -u src/00_smoke_test.py

Part 1 - unit tests:
  Seismic format and CSR, the remap Seismic id -> SAE latent, exact search vs brute force, metrics (vs hand
  computation and ranx), THE WORKED EXAMPLE OF SEC. 3.4 (score 0 -> 0.748), the block-wise M equal to B^T B
  (also after a resume), every normalization and the expansion equal to a dense reference, the pruning rules,
  search of expanded queries vs brute force, LCE bridging a vocabulary mismatch, the val / test split of dev.
Part 2 - end to end: a fake "MS MARCO" with provided SPLARE files in the real formats (large SAE ids,
  alphabetical mapping, string ids, weights rounded to 3 decimals) and scripts 01-08 + the plots, one after the
  other, including the guards of the protocol (the split cannot change, 07 needs the validation, 06 refuses to
  choose again after the test).
"""
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zlib
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

PASSED = []
CPU = torch.device("cpu")


def ok(name):
    PASSED.append(name)
    print(f"  [OK] {name}", flush=True)


# =====================================================================
#  PART 1 — unit tests
# =====================================================================
def test_seismic_and_csr(tmp):
    from sparse_io import CSR, read_seismic_bin, write_seismic_bin
    rng = np.random.default_rng(0)
    dense = (rng.random((50, 40)) < 0.15) * rng.random((50, 40)).astype(np.float32)
    dense[7] = 0  # empty row
    csr = CSR.from_dense(dense)
    write_seismic_bin(tmp / "x.bin", csr)
    back = read_seismic_bin(tmp / "x.bin", verbose=False)
    back.n_cols = 40
    assert np.allclose(back.to_dense(), dense), "roundtrip Seismic"
    part = read_seismic_bin(tmp / "x.bin", max_vectors=10, verbose=False)
    part.n_cols = 40
    assert np.allclose(part.to_dense(), dense[:10])
    rows = np.array([3, 7, 0, 49, 3])
    assert np.allclose(csr.subset(rows).to_dense(), dense[rows]), "subset/gather"
    csr.ids = np.arange(50) + 1000
    csr.save(tmp / "csr")
    again = CSR.load(tmp / "csr", mmap=True)
    assert np.allclose(again.to_dense(), dense) and again.ids[3] == 1003
    ok("Seismic format + CSR (write, read, subsets, save)")


def test_remap():
    from sparse_io import CSR, build_numeric_remap, remap_columns
    sae = sorted(["0", "1", "10", "100", "2", "131071"])          # alphabetical, as in token_to_id_mapping.json
    r2 = build_numeric_remap(sae)
    assert r2.tolist() == [0, 1, 10, 100, 131071, 2]
    csr = CSR.from_dense(np.array([[1, 0, 2, 3, 0, 0], [0, 4, 0, 0, 5, 6]], dtype=np.float32))
    d = remap_columns(csr, r2, 131072).to_dense()
    assert d[0, 0] == 1 and d[0, 10] == 2 and d[0, 100] == 3 and d[1, 1] == 4 and d[1, 131071] == 5 and d[1, 2] == 6
    try:
        build_numeric_remap(["12", "heart"])
        raise AssertionError("a non-numeric key must be rejected")
    except ValueError:
        pass
    ok("remap Seismic id -> SAE latent index (alphabetical mapping)")


def test_search():
    from retrieval import TorchCSR, sparse_search
    from sparse_io import CSR
    rng = np.random.default_rng(1)
    D = (rng.random((300, 60)) < 0.2) * rng.random((300, 60))
    Q = (rng.random((17, 60)) < 0.3) * rng.random((17, 60))
    td = TorchCSR(CSR.from_dense(D), CPU)
    S = Q @ D.T
    s, i = sparse_search(td, CSR.from_dense(Q), 25, q_batch=5, doc_chunk=40, verbose=False)
    assert np.allclose(s, np.sort(S, axis=1)[:, ::-1][:, :25], atol=1e-5), "sparse search scores"
    assert np.allclose(np.take_along_axis(S, i, 1), s, atol=1e-5), "sparse search indices"
    ok("exact sparse search = brute force (also chunked)")


def test_metrics():
    from metrics import evaluate_run
    qrels = {1: {10: 1}, 2: {20: 1, 21: 1}, 3: {30: 3, 31: 1, 32: 2}}
    run = np.array([[5, 10, 6, 7], [20, 9, 8, 21], [31, 30, 4, 32]])
    m, _ = evaluate_run([1, 2, 3], run, qrels, rel_threshold=1, cutoffs_recall=(2, 4), ndcg_cut=3, mrr_cut=3)
    assert abs(m["MRR@3"] - (0.5 + 1 + 1) / 3) < 1e-9
    assert abs(m["R@2"] - (1 + 0.5 + 2 / 3) / 3) < 1e-9
    d = 1 / np.log2(np.arange(2, 6))
    nd3 = (1 * d[0] + 3 * d[1]) / (3 * d[0] + 2 * d[1] + 1 * d[2])
    assert abs(m["nDCG@3"] - (d[1] / d[0] + d[0] / (d[0] + d[1]) + nd3) / 3) < 1e-9
    try:
        from ranx import Qrels, Run, evaluate
        qr = Qrels({str(q): {str(p): r for p, r in v.items()} for q, v in qrels.items()})
        rn = Run({str(q): {str(p): float(10 - j) for j, p in enumerate(row)} for q, row in zip([1, 2, 3], run)})
        rx = evaluate(qr, rn, ["mrr@3", "ndcg@3", "recall@2"])
        assert abs(rx["mrr@3"] - m["MRR@3"]) < 1e-6 and abs(rx["recall@2"] - m["R@2"]) < 1e-6
        assert abs(rx["ndcg@3"] - m["nDCG@3"]) < 1e-6
        ok("metrics (MRR, nDCG, Recall) = hand computation = ranx")
    except ImportError:
        ok("metrics (MRR, nDCG, Recall) = hand computation (ranx not installed)")


def build_cooc(tmp, dense_docs, name, **kw):
    from cooccurrence import Cooccurrence, build_cooccurrence
    from sparse_io import CSR
    docs = CSR.from_dense(dense_docs)
    build_cooccurrence(docs, tmp / name, CPU, **kw)
    return Cooccurrence(tmp / name), docs


def test_paper_example(tmp):
    """Sec. 3.4: 6 documents, 11 features, query F1 0.94 + F4 0.82, alpha 0.5 (absolute), tau 0.5."""
    from lce import expand, expansion_terms, prune
    from sparse_io import CSR
    feats = [[1, 5, 7], [6, 8, 10], [1, 2, 3], [4, 5, 9], [6, 8, 11], [5, 6]]          # DOC-01 .. DOC-06
    X = np.zeros((6, 11), dtype=np.float32)
    for r, fs in enumerate(feats):
        X[r, [f - 1 for f in fs]] = 1.0
    cooc, _ = build_cooc(tmp, X, "paper", row_block=4, doc_chunk=4)
    M = np.asarray(cooc.M)
    assert np.array_equal(M, X.T @ X), "M = B^T B (diagonal = df)"
    assert M[5, 7] == 2 and M[0, 4] == 1                                                  # M[F6,F8]=2, M[F1,F5]=1
    q = np.zeros((1, 11), dtype=np.float32)
    q[0, 0], q[0, 3] = 0.94, 0.82
    E = torch.from_numpy(expansion_terms(CSR.from_dense(q), cooc, ["raw"], CPU, verbose=False)["raw"])
    vq = torch.from_numpy(q)
    vext = expand(vq, E, 0.5, "absolute")
    assert torch.allclose(vext[0], torch.tensor([0.94, 0.47, 0.47, 0.82, 0.88, 0, 0.47, 0, 0.41, 0, 0]), atol=1e-6)
    pruned = prune(vext, vq, "tau:0.5", keep_original=False)
    assert torch.allclose(pruned[0], torch.tensor([0.94, 0, 0, 0.82, 0.88, 0, 0, 0, 0, 0, 0]), atol=1e-6)
    d6 = torch.zeros(11)
    d6[4], d6[5] = 0.85, 0.91
    assert abs(float(vq[0] @ d6)) < 1e-9 and abs(float(pruned[0] @ d6) - 0.748) < 1e-6
    ok("LCE: reproduced EXACTLY the example of Sec. 3.4 (M, V_q_ext, pruning, score 0 -> 0.748)")


def ref_normalize(M, df, n_docs, kind, min_count=1):
    """Dense numpy reference of the four normalizations (independent of lce.py)."""
    A = M.astype(np.float64).copy()
    np.fill_diagonal(A, 0)
    if min_count > 1:
        A[A < min_count] = 0
    d = np.maximum(df.astype(np.float64), 1)
    if kind == "cond":
        A = A / d[:, None]
    elif kind == "cosine":
        A = A / np.sqrt(d[:, None] * d[None, :])
    elif kind == "ppmi":
        with np.errstate(divide="ignore"):
            P = np.log(n_docs * A / (d[:, None] * d[None, :]))
        A = np.where(A > 0, np.maximum(P, 0), 0)
    return A


def test_cooccurrence_and_expansion(tmp):
    from cooccurrence import FILES, Cooccurrence, build_cooccurrence
    from lce import NORMS, expansion_terms
    from sparse_io import CSR
    from utils import load_json, save_json
    rng = np.random.default_rng(3)
    n_feat = 70
    D = (rng.random((230, n_feat)) < 0.12) * (rng.random((230, n_feat)) + 0.1)
    D[:, [5, 33, 69]] = 0                                     # inactive features (df = 0)
    cooc, docs = build_cooc(tmp, D.astype(np.float32), "rand", row_block=9, doc_chunk=17)
    B = (D > 0).astype(np.float64)
    full = B.T @ B
    act = cooc.active
    assert act.tolist() == [f for f in range(n_feat) if f not in (5, 33, 69)]
    assert np.array_equal(np.asarray(cooc.M), full[np.ix_(act, act)]), "block-wise M = B^T B"
    # resume: forget two blocks, rebuild, same matrix
    prog = load_json(tmp / "rand" / FILES["progress"])
    prog["done"] = prog["done"][2:]
    for k in list(prog["stats"])[:2]:
        prog["stats"].pop(k)
    save_json(prog, tmp / "rand" / FILES["progress"])
    m = build_cooccurrence(docs, tmp / "rand", CPU, row_block=9, doc_chunk=17)
    c2 = Cooccurrence(tmp / "rand")
    assert np.array_equal(np.asarray(c2.M), full[np.ix_(act, act)]) and m["complete"], "resume"
    off = full.copy()
    np.fill_diagonal(off, 0)
    assert m["nnz_offdiag"] == int((off[np.ix_(act, act)] > 0).sum()), "statistics of M"
    # expansion = dense reference, for every normalization, with query groups and row blocks
    Q = (rng.random((13, n_feat)) < 0.15) * rng.random((13, n_feat))
    Q[0, 5] = 0.7                                             # a feature absent from the corpus
    qc = CSR.from_dense(Q.astype(np.float32))
    for mc in (1, 3):
        E = expansion_terms(qc, c2, list(NORMS), CPU, min_count=mc, row_block=5,
                            gpu_budget=4 * c2.n_active * 4 * 4 + 3 * 5 * c2.n_active * 4, verbose=False)
        for kind in NORMS:
            A = np.zeros((n_feat, n_feat))
            A[np.ix_(act, act)] = ref_normalize(full[np.ix_(act, act)], full.diagonal()[act], 230, kind, mc)
            assert np.allclose(E[kind], Q @ A, rtol=1e-4, atol=1e-5), (kind, mc)
    ok("M block-wise (and resumed) = B^T B; E = A.V_q for raw / P(j|i) / cosine / PPMI, min count 1 and 3")


def test_prune_rules():
    from lce import LCEConfig, expand, prune
    vq = torch.tensor([[0.0, 2.0, 0.0, 0.1, 0.0, 0.0]])
    E = torch.tensor([[5.0, 1.0, 3.0, 0.0, 1.0, 0.5]])
    assert torch.equal(expand(vq, E, 0.0), vq), "alpha 0 = no expansion"
    v = expand(vq, E, 0.5)                                      # relative: the largest term = 0.5 * max(V_q) = 1
    assert torch.allclose(v, torch.tensor([[1.0, 2.2, 0.6, 0.1, 0.2, 0.1]]))
    assert torch.allclose(prune(v, vq, "add:1"), torch.tensor([[1.0, 2.2, 0, 0.1, 0, 0]])), "add:n"
    assert torch.allclose(prune(v, vq, "rtau:0.15"), torch.tensor([[1.0, 2.2, 0.6, 0.1, 0, 0]])), "rtau keeps originals"
    assert torch.allclose(prune(v, vq, "rtau:0.15", keep_original=False), torch.tensor([[1.0, 2.2, 0.6, 0, 0, 0]]))
    try:
        LCEConfig.parse("cond:0.2:topk:5")
        raise AssertionError("topk must not be accepted")
    except SystemExit:
        pass
    assert torch.allclose(prune(v, vq, "tau:0.5", keep_original=False), torch.tensor([[1.0, 2.2, 0.6, 0, 0, 0]]))
    assert torch.equal(prune(v, vq, "none"), v)
    c = LCEConfig.parse("cond:0.2:add:20")
    assert c.name() == "lce_cond_a0.2_add20" and LCEConfig.from_dict(c.to_dict()) == c
    assert LCEConfig(None, 0.0).name() == "splare" and LCEConfig.from_dict({"norm": float("nan")}).is_baseline
    ok("pruning rules (none, tau, rtau, add; no top-k), relative alpha, configuration names")


def test_src_sources(tmp):
    """Second round: expanding only from the strong latents of the query (src). src = 0 must change NOTHING of the
    first round (names, grid, files, numbers); src > 0 must equal a dense reference computed by hand."""
    from cooccurrence import Cooccurrence
    from lce import NORMS, LCEConfig, build_grid, compute_expansions, expansion_terms
    from sparse_io import CSR
    # --- nothing of the first round changes
    g = build_grid(list(NORMS), [0.05, 0.1, 0.2, 0.3, 0.5, 1.0],
                   ["add:5", "add:10", "add:20", "add:50", "add:100", "rtau:0.05", "rtau:0.1"])
    assert len(g) == 169 and g[0].is_baseline and all(c.src == 0.0 for c in g)
    assert LCEConfig.parse("cond:0.2:add:20").name() == "lce_cond_a0.2_add20"
    old = {"norm": "cosine", "alpha": 0.2, "prune": "add:20", "alpha_mode": "relative", "keep_original": True,
           "min_count": 1}                                                  # lce_best.json written by the first round
    c = LCEConfig.from_dict(old)
    assert c.src == 0.0 and c.name() == "lce_cosine_a0.2_add20" and c.e_key == ("cosine", 1, 0.0)
    assert LCEConfig.from_dict({**old, "src": float("nan")}).src == 0.0
    c5 = LCEConfig.parse("cond:0.2:add:20:src0.5")
    assert c5.src == 0.5 and c5.name() == "lce_cond_a0.2_add20_src0.5" and LCEConfig.from_dict(c5.to_dict()) == c5
    assert LCEConfig.parse("cond:0.2:add:20:mc3:src0.5").e_key == ("cond", 3, 0.5)
    assert LCEConfig.parse("cond:0.2:src0.25").prune == "none"
    for bad in ("cond:0.2:add:20:src1.5", "cond:0.2:add:20:src-0.1"):
        try:
            LCEConfig.parse(bad)
            raise AssertionError(bad)
        except SystemExit:
            pass
    g2 = build_grid(["cond"], [0.1, 0.2], ["add:5"], srcs=(0.0, 0.5))
    assert [x.name() for x in g2] == ["splare", "lce_cond_a0.1_add5", "lce_cond_a0.2_add5",
                                      "lce_cond_a0.1_add5_src0.5", "lce_cond_a0.2_add5_src0.5"]
    # --- the expansion from the strong latents = dense reference
    rng = np.random.default_rng(8)
    n_feat = 60
    D = (rng.random((200, n_feat)) < 0.13) * (rng.random((200, n_feat)) + 0.1)
    D[:, [4, 20]] = 0
    cooc, _ = build_cooc(tmp, D.astype(np.float32), "srcm", row_block=9, doc_chunk=17)
    c2 = Cooccurrence(tmp / "srcm")
    act = c2.active
    full = (D > 0).astype(np.float64).T @ (D > 0).astype(np.float64)
    Q = (rng.random((11, n_feat)) < 0.2) * (rng.random((11, n_feat)) + 0.2)
    Q[0] = 0
    Q[0, [3, 7]] = [0.5, 1.0]                                              # weights exactly at 0.5 x max
    Q[1] = 0                                                                # an empty query
    Q = Q.astype(np.float32)
    qc = CSR.from_dense(Q)
    for src in (0.0, 0.5, 1.0):
        E = expansion_terms(qc, c2, ["raw", "cosine"], CPU, row_block=5, verbose=False, src=src)
        Qs = np.where((Q > 0) & (Q >= np.float32(src) * Q.max(axis=1, keepdims=True)), Q, 0)
        for kind in ("raw", "cosine"):
            A = np.zeros((n_feat, n_feat))
            A[np.ix_(act, act)] = ref_normalize(full[np.ix_(act, act)], full.diagonal()[act], 200, kind)
            assert np.allclose(E[kind], Qs @ A, rtol=1e-4, atol=1e-5), (kind, src)
        if src == 0.5:
            assert Qs[0, 3] > 0 and Qs[0, 7] > 0, "a weight equal to src x max is kept (>=)"
    # src = 1 keeps only the strongest latent of each query; src = 0 equals the default call
    assert np.array_equal(expansion_terms(qc, c2, ["raw"], CPU, row_block=5, verbose=False, src=0.0)["raw"],
                          expansion_terms(qc, c2, ["raw"], CPU, row_block=5, verbose=False)["raw"])
    # --- compute_expansions keys: one E per (norm, min_count, src)
    cfgs = [LCEConfig("raw", 0.2, "add:5"), LCEConfig("raw", 0.2, "add:5", src=0.5), LCEConfig("cosine", 0.1, "add:5")]
    EE = compute_expansions(qc, c2, cfgs, CPU, 5)
    assert set(EE) == {("raw", 1, 0.0), ("raw", 1, 0.5), ("cosine", 1, 0.0)}
    assert not np.array_equal(EE[("raw", 1, 0.0)], EE[("raw", 1, 0.5)]), "src must change the expansion"
    ok("src: first round unchanged (names, grid of 169, old lce_best.json); expansion from strong latents = dense reference")


def test_lce_bridges_mismatch(tmp):
    """Lay query features never appear in the relevant documents (vocabulary mismatch); other documents use
    both vocabularies. Without expansion the relevant document has score 0; LCE must bring it up."""
    from lce import LCEConfig, compute_expansions, expand, prune, search_configs
    from metrics import evaluate_run
    from retrieval import TorchCSR
    from sparse_io import CSR
    rng = np.random.default_rng(5)
    T, n_lay, n_pro = 12, 6, 6
    n_feat = T * (n_lay + n_pro) + 40
    lay = lambda t: t * (n_lay + n_pro) + np.arange(n_lay)                  # noqa: E731
    pro = lambda t: t * (n_lay + n_pro) + n_lay + np.arange(n_pro)          # noqa: E731
    noise = np.arange(T * (n_lay + n_pro), n_feat)
    docs, qrows, qrels = [], [], {}
    for t in range(T):
        for _ in range(10):                                  # bridge documents: lay + professional
            v = np.zeros(n_feat)
            v[rng.choice(lay(t), 3, replace=False)] = rng.random(3) + 0.5
            v[rng.choice(pro(t), 3, replace=False)] = rng.random(3) + 0.5
            v[rng.choice(noise, 4, replace=False)] = rng.random(4) + 0.5
            docs.append(v)
    for t in range(T):
        for _ in range(4):                                   # relevant documents: professional only
            v = np.zeros(n_feat)
            v[rng.choice(pro(t), 4, replace=False)] = rng.random(4) + 1.0
            v[rng.choice(noise, 2, replace=False)] = rng.random(2) + 0.2
            docs.append(v)
            q = np.zeros(n_feat)
            q[rng.choice(lay(t), 3, replace=False)] = rng.random(3) + 0.5
            qrels[len(qrows)] = {len(docs) - 1: 1}
            qrows.append(q)
    Dm, Qm = np.array(docs, dtype=np.float32), np.array(qrows, dtype=np.float32)
    cooc, dcsr = build_cooc(tmp, Dm, "bridge")
    q = CSR.from_dense(Qm, ids=np.arange(len(qrows)))
    base, cfg = LCEConfig(None, 0.0), LCEConfig("cond", 1.0, "add:6")
    E = compute_expansions(q, cooc, [cfg], CPU, 64)
    res = search_configs(TorchCSR(dcsr, CPU), q, E, [base, cfg], 20, CPU, 16, verbose=False)
    m0, _ = evaluate_run(np.arange(len(qrows)), res[base]["rows"], qrels)
    m1, _ = evaluate_run(np.arange(len(qrows)), res[cfg]["rows"], qrels)
    # the expanded scores must equal a brute-force dot product with the same expanded vectors
    vq = torch.from_numpy(Qm)
    vext = prune(expand(vq, torch.from_numpy(E[cfg.e_key]), 1.0), vq, "add:6").numpy()
    S = vext @ Dm.T
    assert np.allclose(np.take_along_axis(S, res[cfg]["rows"], 1), res[cfg]["scores"], atol=1e-4)
    assert res[cfg]["avg_added"] <= 6 + 1e-9
    assert m1["R@10"] > m0["R@10"] + 0.3, (m0["R@10"], m1["R@10"])
    ok(f"LCE bridges a vocabulary mismatch (R@10 {m0['R@10']:.2f} -> {m1['R@10']:.2f}; MRR@10 "
       f"{m0['MRR@10']:.2f} -> {m1['MRR@10']:.2f}); expanded scores = brute force")


def test_split():
    """val / test of dev: disjoint, complete, deterministic, equal to the rule written out by hand."""
    from data import compute_split
    rng = np.random.default_rng(11)
    qids = rng.choice(10 ** 7, 500, replace=False)
    qrels = {int(q): {int(rng.integers(10 ** 6)): 1} for q in qids[:480]}             # 20 without judgments
    qrels[123] = {5: 1}                                                                # judged, no vector
    qrels[int(qids[0])] = {7: 0}                                                       # judged, nothing relevant
    val, test, usable = compute_split(qids, qrels, n_val=100, seed=42)
    assert len(usable) == 479 and len(val) == 100 and len(test) == 379
    assert len(np.intersect1d(val, test)) == 0 and np.array_equal(np.union1d(val, test), usable)
    assert np.all(np.diff(val) > 0) and np.all(np.diff(test) > 0)
    ref = np.sort(np.array([q for q in qids.tolist()[1:480]], dtype=np.int64))      # the rule, by hand
    perm = np.random.default_rng(42).permutation(len(ref))
    assert np.array_equal(val, np.sort(ref[perm[:100]])) and np.array_equal(test, np.sort(ref[perm[100:]]))
    v2, t2, _ = compute_split(qids[::-1], qrels, n_val=100, seed=42)                  # order of input irrelevant
    assert np.array_equal(v2, val) and np.array_equal(t2, test)
    ok("val / test split of dev: disjoint, complete, sorted, deterministic (seed 42)")


# =====================================================================
#  PART 2 — synthetic world + scripts 01..08
# =====================================================================
def bf16ish(x):
    """Weights with many binary digits, like the bfloat16 values of real SPLARE vectors (1.5078125)."""
    return np.round(np.asarray(x, dtype=np.float64) * 128) / 128 + 1 / 1024


def build_world(root, n_docs=400, n_dev=80):
    from sparse_io import CSR, write_seismic_bin
    rng = np.random.default_rng(42)
    words = [f"w{i}" for i in range(120)]

    # --- texts: 10 topics x 12 words
    def text(topic, n):
        return " ".join(rng.choice(words[topic * 12:(topic + 1) * 12], n).tolist() + rng.choice(words, 2).tolist())
    doc_topic = rng.integers(0, 10, n_docs)
    collection = [text(t, 12) for t in doc_topic]
    dev_qids = np.sort(rng.choice(np.arange(10 ** 6, 2 * 10 ** 6), n_dev, replace=False))
    pos = rng.choice(n_docs, n_dev, replace=False)
    dv_t = {int(q): text(int(doc_topic[p]), 3) for q, p in zip(dev_qids, pos)}
    dv_r = {int(q): {int(p): 1} for q, p in zip(dev_qids, pos)}
    dv_r[int(dev_qids[3])][int((pos[3] + 1) % n_docs)] = 1                  # a query with two relevant passages

    msm = root / "msm_src"
    msm.mkdir()
    (msm / "collection.tsv").write_text("".join(f"{i}\t{t}\n" for i, t in enumerate(collection)))
    (msm / "queries.dev.small.tsv").write_text("".join(f"{q}\t{s}\n" for q, s in dv_t.items()))
    (msm / "qrels.dev.small.tsv").write_text("".join(f"{q}\t0\t{p}\t{v}\n" for q, d in dv_r.items()
                                                     for p, v in d.items()))
    with tarfile.open(root / "collectionandqueries.tar.gz", "w:gz") as tar:
        for f in msm.iterdir():
            tar.add(f, arcname=f.name)

    # --- fake "SAE": LARGE latent ids, 3 latents per word + 1 per topic
    lat_pool = rng.choice(131072, 3 * len(words) + 10, replace=False)
    word_lat = {w: lat_pool[3 * i:3 * i + 3] for i, w in enumerate(words)}
    topic_lat = {t: lat_pool[3 * len(words) + t] for t in range(10)}
    lat_w = {int(l): float(w) for l, w in zip(lat_pool, bf16ish(rng.random(len(lat_pool)) * 2 + 0.3))}

    def splare(texts):
        """Deterministic fake encoder: the weight of a latent depends on the word AND on the text (context),
        so that, as with the real encoder, exact score ties are rare."""
        rows = []
        for t in texts:
            ctx = np.random.default_rng(zlib.crc32(t.encode()))
            v = {}
            for w in t.split():
                for latent in list(word_lat[w]) + [topic_lat[words.index(w) // 12]]:
                    x = float(bf16ish(lat_w[int(latent)] * (0.7 + 0.6 * ctx.random())))
                    v[int(latent)] = max(v.get(int(latent), 0.0), x)
            rows.append(v)
        return rows

    def to_csr(rows):
        dense = np.zeros((len(rows), 131072), dtype=np.float32)
        for i, v in enumerate(rows):
            for latent, w in v.items():
                dense[i, latent] = w
        return CSR.from_dense(dense)

    # --- provided files: alphabetical mapping, ids as '<U7' strings, weights rounded to 3 decimals
    perm = rng.permutation(n_docs)
    dcsr = to_csr(splare([collection[i] for i in perm]))
    qcsr = to_csr(splare([dv_t[int(q)] for q in dev_qids]))
    folder = root / "splare_provided"
    folder.mkdir()
    used = np.union1d(np.unique(dcsr.indices), np.unique(qcsr.indices))
    keys = sorted(str(int(c)) for c in used)
    mapping = {k: i for i, k in enumerate(keys)}
    lut = np.full(131072, -1, dtype=np.int64)
    for k, i in mapping.items():
        lut[int(k)] = i
    rounded = {}
    for name, c in [("documents.bin", dcsr), ("queries.bin", qcsr)]:
        rounded[name] = CSR(c.indptr, c.indices, np.round(c.data.astype(np.float64), 3).astype(np.float32), c.n_cols)
        write_seismic_bin(folder / name, CSR(c.indptr, lut[c.indices].astype(np.int32), rounded[name].data,
                                             len(mapping)))
    np.save(folder / "doc_ids.npy", np.array([str(p) for p in perm], dtype="<U7"))
    np.save(folder / "queries_ids.npy", np.array([str(q) for q in dev_qids], dtype="<U7"))
    (folder / "token_to_id_mapping.json").write_text(json.dumps(mapping))
    S = rounded["queries.bin"].to_dense() @ rounded["documents.bin"].to_dense().T
    for fname, kk in [("groundtruth_100.tsv", 100), ("groundtruth.tsv", 10)]:
        with open(folder / fname, "w") as f:
            for qi in range(S.shape[0]):
                for r, di in enumerate(np.argsort(-S[qi], kind="stable")[:kk]):
                    f.write(f"{qi}\t{di}\t{r + 1}\t{S[qi, di]:.5f}\n")
    return {"n_dev": n_dev}


def run(script, *args, env, expect_fail=False):
    path = Path(script) if Path(script).is_absolute() else HERE / script
    cmd = [sys.executable, "-u", str(path), *map(str, args)]
    print(f"\n$ {Path(script).name} {' '.join(map(str, args))}" + ("   (must fail)" if expect_fail else ""), flush=True)
    p = subprocess.run(cmd, env=env, cwd=str(HERE), capture_output=True, text=True)
    text = (p.stdout + p.stderr).strip()
    tail = "\n".join(text.splitlines()[-4:])
    print("    " + tail.replace("\n", "\n    "), flush=True)
    if expect_fail:
        if p.returncode == 0:
            raise SystemExit(f"FAILED: {script} {args} should have stopped with an error")
        return text
    if p.returncode != 0:
        print(p.stdout[-5000:])
        print(p.stderr[-5000:])
        raise SystemExit(f"FAILED: {script}")
    return p.stdout


def end_to_end(root):
    w = build_world(root)
    work = root / "work"
    env = dict(os.environ, SPLARE_DIR=str(root / "splare_provided"), WORK_DIR=str(work), CUDA_VISIBLE_DEVICES="")
    res_dir = work / "results"
    load = lambda name: json.loads((res_dir / f"{name}.json").read_text())  # noqa: E731

    run("01_prepare_data.py", env=env)
    run("02_download_msmarco.py", "--from-tar", root / "collectionandqueries.tar.gz", env=env)

    # the split of dev, and its guards
    n_val = 30
    run("03_split_dev.py", "--n-val", n_val, env=env)
    out = run("03_split_dev.py", "--n-val", n_val, env=env)
    assert "already exists and is identical" in out
    run("03_split_dev.py", "--n-val", 20, env=env, expect_fail=True)
    val = np.loadtxt(work / "splits" / "val_qids.txt", dtype=np.int64)
    test = np.loadtxt(work / "splits" / "test_qids.txt", dtype=np.int64)
    assert len(val) == n_val and len(test) == w["n_dev"] - n_val and len(np.intersect1d(val, test)) == 0
    ok(f"split: {len(val)} val + {len(test)} test, disjoint; rerun = no-op; a different split is refused")

    # SPLARE alone, and the check against the provided ground truth
    run("04_baseline.py", "--k", 50, env=env)
    dev = load("splare_dev")["summary"]
    assert dev["overlap_groundtruth_100"] > 0.99 and dev["overlap_groundtruth"] > 0.99, dev
    assert load("splare_val")["summary"]["n_queries"] == n_val
    assert load("splare_test")["summary"]["n_queries"] == len(test)
    pq_v, pq_t = load("splare_val")["per_query"], load("splare_test")["per_query"]
    assert not set(pq_v["qid"]) & set(pq_t["qid"])
    assert sorted(pq_v["qid"] + pq_t["qid"]) == sorted(load("splare_dev")["per_query"]["qid"])
    ok("baseline: overlap with the provided ground truth = 1.00; val + test = dev, disjoint")

    # LCE: offline, validation, test
    run("05_build_cooccurrence.py", "--row-block", 50, "--doc-chunk", 64, env=env)
    run("07_test_lce.py", "--k", 50, env=env, expect_fail=True)                 # no validation yet
    grid = ["--norms", "raw", "cond", "ppmi", "--alphas", 0.2, 1.0, "--prunes", "add:5", "rtau:0.05",
            "--k", 50, "--q-batch", 16, "--row-block", 40]
    out = run("06_validate_lce.py", *grid, env=env)
    assert f"validation: {n_val} queries, 13 configurations" in out
    assert not list(res_dir.glob("lce_*_test.json")), "06 must not touch the test set"
    best = load("lce_best")
    assert best["selected_on"] == "val" and best["n_queries"] == n_val
    run("07_test_lce.py", "--k", 50, "--q-batch", 16, env=env)
    lce = load(f"{best['name']}_test")["summary"]
    assert lce["n_queries"] == len(test) and lce["selection"].startswith("validation")
    assert abs(load("splare_test")["summary"]["MRR@10"] - lce["baseline_MRR@10"]) < 1e-9
    an = load("lce_test_analysis")
    assert an["MRR@10"]["wins"] + an["MRR@10"]["ties"] + an["MRR@10"]["losses"] == len(test)
    assert (res_dir / "lce_examples_test.txt").exists() and (work / "runs" / f"{best['name']}_test.trec").exists()
    run("06_validate_lce.py", *grid, env=env, expect_fail=True)                  # test already seen
    ok(f"LCE: grid on val only ({best['name']} chosen), tested once on test; 06 refuses to choose again")

    run("07_test_lce.py", "--config", "cosine:0.3:add:10", "--k", 50, "--q-batch", 16, env=env)
    assert load("lce_cosine_a0.3_add10_test")["summary"]["selection"].startswith("manual")
    assert load("lce_test_analysis")["config"] == best["name"], "an ablation must not replace the main analysis"
    # a SECOND, declared round of validation (src): the files of the first round must stay exactly as they are
    keep_files = ["lce_validation.csv", "lce_best.json", "lce_validation_best_per_norm.csv",
                  "lce_test_analysis.json", "lce_examples_test.txt", f"{best['name']}_test.json"]
    first = {n: (res_dir / n).read_bytes() for n in keep_files}
    g2 = ["--norms", "raw", "cond", "--alphas", 0.2, 1.0, "--prunes", "add:5", "--k", 50, "--q-batch", 16,
          "--row-block", 40]
    run("06_validate_lce.py", *g2, "--src", 0, 0.5, "--tag", "chk", env=env, expect_fail=True)         # test seen
    run("06_validate_lce.py", *g2, "--src", 0, 0.5, "--after-test", env=env, expect_fail=True)         # no --tag
    out = run("06_validate_lce.py", *g2, "--src", 0, 0.5, "--tag", "chk", "--after-test", env=env)
    assert "SECOND ROUND" in out and f"validation: {n_val} queries, 9 configurations" in out
    # 5 configurations of the first round are re-run (SPLARE + raw/cond x alpha 0.2/1 x add:5): same numbers, exactly
    assert "check against the first round: 5 configurations in common" in out
    assert "max |MRR@10 difference| = 0.000000" in out, out[-1500:]
    out = run("06_validate_lce.py", *g2, "--src", 0.5, 0.75, "--tag", "src", "--after-test", env=env)
    assert "Best configuration for each value of src" in out
    b2 = load("lce_best_src")
    assert b2["config"]["src"] in (0.5, 0.75) and "_src" in b2["name"] and b2["tag"] == "src"
    assert (res_dir / "lce_validation_src.csv").exists() and (res_dir / "lce_validation_best_per_norm_src.csv").exists()
    run("07_test_lce.py", "--tag", "src", "--k", 50, "--q-batch", 16, env=env)
    t2 = load(f"{b2['name']}_test")["summary"]
    assert "second round" in t2["selection"] and t2["n_queries"] == len(test) and t2["src"] == b2["config"]["src"]
    assert (res_dir / "lce_test_analysis_src.json").exists() and (res_dir / "lce_examples_test_src.txt").exists()
    assert load("lce_test_analysis_src")["config"] == b2["name"]
    run("07_test_lce.py", "--tag", "nothing", "--k", 50, env=env, expect_fail=True)             # no such round
    shutil.copy(res_dir / "lce_best.json", res_dir / "lce_best_same.json")                     # same choice as round 1
    out = run("07_test_lce.py", "--tag", "same", "--k", 50, env=env)
    assert "chose the same configuration as the first round" in out
    for n, content in first.items():
        assert (res_dir / n).read_bytes() == content, f"{n} was modified by the second round"
    out = run("08_significance.py", "--iters", 200, "--tag", "src", env=env)
    assert b2["name"] in out
    ok(f"second round (src): declared with --tag/--after-test (refused otherwise), identical numbers for the "
       f"configurations of round 1, files of round 1 untouched; chosen {b2['name']}")

    out = run("08_significance.py", "--iters", 500, env=env)
    assert f"paired queries: {len(test)}" in out and best["name"] in out
    plots = HERE.parent / "plots" / "lce_plots.py"
    run(plots, "--results", res_dir, env=env)
    figs = HERE.parent / "plots" / "figures"
    assert (figs / "fig1_validation.png").exists(), "fig1_validation"
    assert list(figs.glob("fig[2-5]_test_*.png")), "no fig2..fig5"
    assert "SPLARE + LCE" in (figs / "tables.md").read_text()
    summary = (res_dir / "summary.csv").read_text().splitlines()
    ok(f"end to end: scripts 01-08, the plots and the tables, {len(summary) - 1} rows in results/summary.csv")


def main():
    print("PART 1: unit tests", flush=True)
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        test_seismic_and_csr(tmp)
        test_remap()
        test_search()
        test_metrics()
        test_paper_example(tmp)
        test_cooccurrence_and_expansion(tmp)
        test_prune_rules()
        test_src_sources(tmp)
        test_lce_bridges_mismatch(tmp)
        test_split()
    print("\nPART 2: full pipeline on synthetic data", flush=True)
    keep = "--keep" in sys.argv
    root = Path(tempfile.mkdtemp(prefix="splare_lce_smoke_"))
    figs = HERE.parent / "plots" / "figures"
    before = {p.name for p in figs.glob("*")} if figs.exists() else set()
    try:
        end_to_end(root)
    finally:
        # the smoke test must not leave fake figures in plots/figures/
        if figs.exists():
            for p in figs.glob("*"):
                if p.name not in before:
                    p.unlink()
            if not any(figs.iterdir()) and not before:
                figs.rmdir()
        if keep:
            print(f"(synthetic files left in {root})")
        else:
            shutil.rmtree(root, ignore_errors=True)
    print(f"\nALL TESTS PASSED ({len(PASSED)})", flush=True)


if __name__ == "__main__":
    main()
