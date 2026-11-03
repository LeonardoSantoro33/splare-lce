"""
lce.py — Latent Concept Expansion (Sec. 3 of the paper) in the SPLARE space (131,072 SAE latents).

  Eq. (1)  M_ij = sum_d 1[V_d(i) > 0] 1[V_d(j) > 0],  M_ii = 0          (cooccurrence.py, built offline)
  Eq. (2)  V_q_ext = V_q + alpha * (A . V_q)                            A = M or a normalization of M
  Eq. (3)  pruning of V_q_ext to restore sparsity

NORMALIZATIONS of A (chosen on the validation queries like alpha and the pruning; "raw" is Eq. 1 as written):
  raw     A_ij = M_ij
  cond    A_ij = M_ij / df_i                    = P(j | i), "how often j fires when i fires"
  cosine  A_ij = M_ij / sqrt(df_i df_j)
  ppmi    A_ij = max(0, log(N M_ij / (df_i df_j)))        (0 where M_ij = 0)
  E_j = sum_i V_q(i) A_ij  is computed with the rows of M of the query features only, which is exactly
  (A . V_q): the other rows are multiplied by 0. M itself is never modified.

ALPHA. On 8.8M documents the raw counts reach millions, so with alpha "absolute" (Eq. 2 literally) the
expansion swamps V_q for any usable alpha. Default "relative": per query, alpha_q = alpha * max(V_q) / max(E),
i.e. the strongest expansion term gets alpha times the strongest original weight. It is scale-free, so the
same alpha means the same thing for every normalization and every corpus.

PRUNING (Eq. 3), specs:
  none       keep everything
  tau:x      keep weights > x                       (Eq. 3 literally, absolute threshold)
  rtau:x     keep weights > x * max(V_q)            (Eq. 3 with a scale-free threshold)
  add:n      keep the original features + the n largest ADDED ones (fixed expansion budget)
By default the ORIGINAL query features are never pruned (--keep-original): then alpha = 0 is exactly the
un-expanded model, and LCE can only add concepts and re-weight the original ones, never delete them.

SOURCES (src, optional, default 0 = every latent of the query, i.e. Eq. 2 as above). With src = s > 0 only the
query latents with weight >= s * max(V_q) are used to compute the expansion terms E = A . V_q: the weak latents
of the query no longer "vote" for added concepts. The query itself is NOT changed (all its latents stay in V_q
and in the score), only the terms that are added are computed from fewer latents. s is a fraction of the strongest
weight of the query (a relative threshold, no top-k).
"""
from dataclasses import dataclass

import numpy as np
import torch

from retrieval import csr_rows_to_dense, search_with_query_table
from utils import log, timer

NORMS = ("raw", "cond", "cosine", "ppmi")


# ----------------------------------------------------------------------
#  configurations
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class LCEConfig:
    norm: str                  # raw | cond | cosine | ppmi   (None = no expansion: the baseline)
    alpha: float
    prune: str = "none"
    alpha_mode: str = "relative"
    keep_original: bool = True
    min_count: int = 1
    src: float = 0.0           # 0 = every query latent feeds the expansion; s > 0: only weights >= s * max(V_q)

    @property
    def is_baseline(self):
        return self.norm is None or self.alpha == 0

    @property
    def e_key(self):
        """Key of the expansion terms E = A . V_q this configuration needs (the same E serves many alphas/prunings)."""
        return (self.norm, self.min_count, self.src)

    def name(self):
        """File-name of the system: 'splare' (no expansion) or e.g. 'lce_cond_a0.2_add20'."""
        if self.is_baseline:
            return "splare"
        s = f"lce_{self.norm}_a{self.alpha:g}_{self.prune.replace(':', '')}"
        if self.min_count > 1:
            s += f"_mc{self.min_count}"
        if self.src > 0:
            s += f"_src{self.src:g}"
        if self.alpha_mode != "relative":
            s += "_abs"
        if not self.keep_original:
            s += "_noorig"
        return s

    def to_dict(self):
        return {"norm": self.norm, "alpha": self.alpha, "prune": self.prune, "alpha_mode": self.alpha_mode,
                "keep_original": self.keep_original, "min_count": self.min_count, "src": self.src}

    @staticmethod
    def from_dict(d):
        norm = d.get("norm")
        if norm is None or (isinstance(norm, float) and norm != norm):      # NaN from a csv = the baseline
            return LCEConfig(None, 0.0)
        src = d.get("src", 0.0)                                              # files of the first round have no src
        src = 0.0 if src is None or src != src else float(src)
        return LCEConfig(norm, float(d["alpha"]), d.get("prune", "none"), d.get("alpha_mode", "relative"),
                         bool(d.get("keep_original", True)), int(d.get("min_count", 1)), check_src(src))

    @staticmethod
    def parse(spec, alpha_mode="relative", keep_original=True):
        """'cond:0.2:add:20', 'cond:0.2:add:20:mc5' or 'cond:0.2:add:20:src0.5' -> LCEConfig."""
        parts = spec.split(":")
        if len(parts) < 2 or parts[0] not in NORMS:
            raise SystemExit(f"[ERROR] --config {spec}: expected NORM:ALPHA[:PRUNE[:ARG]][:mcN][:srcS], "
                             f"NORM in {NORMS}")
        mc, src = 1, 0.0
        while len(parts) > 2 and (parts[-1].startswith("mc") or parts[-1].startswith("src")):
            tok = parts.pop()
            if tok.startswith("mc"):
                mc = int(tok[2:])
            else:
                src = check_src(float(tok[3:]))
        prune = ":".join(parts[2:]) if len(parts) > 2 else "none"
        check_prune(prune)
        return LCEConfig(parts[0], float(parts[1]), prune, alpha_mode, keep_original, mc, src)


def check_src(s):
    """src is a fraction of the strongest query weight: 0 (every latent) <= s <= 1."""
    if not 0.0 <= float(s) <= 1.0:
        raise SystemExit(f"[ERROR] src must be between 0 (every query latent) and 1, got {s}")
    return float(s)


def check_prune(spec):
    kind = spec.split(":")[0]
    if spec == "none":
        return spec
    if kind not in ("tau", "rtau", "add") or len(spec.split(":")) != 2:
        raise SystemExit(f"[ERROR] unknown pruning '{spec}': use none, tau:x, rtau:x or add:n")
    float(spec.split(":")[1])
    return spec


def build_grid(norms, alphas, prunes, min_counts=(1,), alpha_mode="relative", keep_original=True, srcs=(0.0,)):
    """SPLARE alone first, then every combination. With the defaults (srcs = (0,)) the grid is exactly the one of
    the first round: src = 0 changes nothing."""
    grid = [LCEConfig(None, 0.0)]
    for mc in min_counts:
        for src in srcs:
            for norm in norms:
                for a in alphas:
                    for p in prunes:
                        grid.append(LCEConfig(norm, float(a), check_prune(p), alpha_mode, keep_original, int(mc),
                                              check_src(src)))
    return grid


# ----------------------------------------------------------------------
#  Eq. (1) -> A: normalization of rows of M
# ----------------------------------------------------------------------
@torch.no_grad()
def normalize_rows(R, df_rows, df_cols, n_docs, kind):
    """R: (b, n) rows of M with ZERO diagonal (counts). df_rows (b,), df_cols (n,). Returns a NEW tensor."""
    if kind == "raw":
        return R.clone()
    dr = df_rows.clamp_min(1.0)[:, None]
    dc = df_cols.clamp_min(1.0)[None, :]
    if kind == "cond":
        return R / dr
    if kind == "cosine":
        return R / dr.sqrt() / dc.sqrt()
    if kind == "ppmi":
        A = R * float(n_docs)
        A.div_(dr).div_(dc)
        A.clamp_(min=1e-30).log_()
        A.mul_(R > 0).clamp_(min=0.0)
        return A
    raise ValueError(f"unknown normalization: {kind}")


# ----------------------------------------------------------------------
#  Eq. (2): expansion terms E = A . V_q for a whole query set
# ----------------------------------------------------------------------
@torch.no_grad()
def expansion_terms(q_csr, cooc, norms, dev, min_count=1, row_block=2048, gpu_budget=None, verbose=True, src=0.0):
    """E[norm] (nq x n_cols, float32 numpy, CPU) for every normalization in `norms`.
    Only the rows of M of the features present in the queries are read (from the memmap, in blocks).
    src > 0: only the query latents with weight >= src * max(V_q) (of that query) feed the expansion."""
    if q_csr.n_cols > cooc.n_cols:
        raise SystemExit(f"[ERROR] queries have {q_csr.n_cols} columns, M has {cooc.n_cols}")
    check_src(src)
    nq = q_csr.n_rows
    rows_all = q_csr.row_ids_per_element()
    cols_all = np.asarray(q_csr.indices, dtype=np.int64)
    vals_all = np.asarray(q_csr.data, dtype=np.float32)
    comp_all = cooc.col_map[cols_all]
    keep = (vals_all > 0) & (comp_all >= 0)          # features absent from the corpus have an empty row in M
    if src > 0:
        qmax = np.zeros(nq, dtype=np.float32)        # max(V_q) of the WHOLE query, as in the relative pruning
        np.maximum.at(qmax, rows_all, vals_all)
        keep &= vals_all >= np.float32(src) * qmax[rows_all]
    qr, qc, qv = rows_all[keep], comp_all[keep], vals_all[keep]

    if gpu_budget is None:
        gpu_budget = torch.cuda.mem_get_info()[0] * 0.45 if dev.type == "cuda" else 4e9
    # full float32 here (TF32 would keep only ~3 significant digits of the weights of A)
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        return _expansion_groups(qr, qc, qv, nq, cooc, norms, dev, min_count, row_block, gpu_budget, verbose)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32


def _expansion_groups(qr, qc, qv, nq, cooc, norms, dev, min_count, row_block, gpu_budget, verbose):
    n = cooc.n_active
    per_query = n * 4 * len(norms)
    group = int(max(1, min(nq, (gpu_budget - 3 * row_block * n * 4) // per_query)))
    df_cols = torch.from_numpy(cooc.df_active.astype(np.float32)).to(dev)
    out = {norm: np.zeros((nq, cooc.n_cols), dtype=np.float32) for norm in norms}
    for g0 in range(0, nq, group):
        g1 = min(nq, g0 + group)
        sel = (qr >= g0) & (qr < g1)
        gr, gc, gv = qr[sel] - g0, qc[sel], qv[sel]
        feats = np.unique(gc)
        pos = np.full(n, -1, dtype=np.int64)
        pos[feats] = np.arange(len(feats))
        E = {norm: torch.zeros(g1 - g0, n, dtype=torch.float32, device=dev) for norm in norms}
        n_blocks = (len(feats) + row_block - 1) // row_block
        if verbose:
            log(f"  expansion: queries {g0:,}-{g1:,}, {len(feats):,} distinct features -> {n_blocks} blocks of rows of M")
        for bi, b0 in enumerate(range(0, len(feats), row_block)):
            fb = feats[b0:b0 + row_block]
            R = torch.from_numpy(cooc.rows(fb)).to(dev)
            R[torch.arange(len(fb), device=dev), torch.from_numpy(fb).to(dev)] = 0.0      # Eq. (1): i != j
            if min_count > 1:
                R[R < min_count] = 0.0
            m = (pos[gc] >= b0) & (pos[gc] < b0 + len(fb))
            Qb = torch.zeros(g1 - g0, len(fb), dtype=torch.float32, device=dev)
            Qb[torch.from_numpy(gr[m]).to(dev), torch.from_numpy(pos[gc[m]] - b0).to(dev)] = \
                torch.from_numpy(gv[m]).to(dev)
            df_rows = torch.from_numpy(cooc.df_active[fb].astype(np.float32)).to(dev)
            for norm in norms:
                A = normalize_rows(R, df_rows, df_cols, cooc.n_docs, norm)
                E[norm].addmm_(Qb, A)
                del A
            del R, Qb
            if verbose and n_blocks >= 10 and bi % max(1, n_blocks // 5) == 0:
                log(f"    rows of M: block {bi + 1}/{n_blocks}")
        for norm in norms:
            out[norm][g0:g1, cooc.active] = E[norm].cpu().numpy()
        del E
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    return out


def compute_expansions(q, cooc, configs, dev, row_block):
    """{(norm, min_count, src): E} for the configurations that need it (LCEConfig.e_key)."""
    need = {}
    for c in configs:
        if not c.is_baseline:
            need.setdefault((c.min_count, c.src), set()).add(c.norm)
    E = {}
    for (mc, src), norms in need.items():
        norms = [n for n in NORMS if n in norms]
        extra = f"; sources >= {src:g} x max V_q" if src > 0 else ""
        with timer(f"expansion terms A.V_q for {q.n_rows:,} queries ({', '.join(norms)}; min count {mc}{extra})"):
            for norm, arr in expansion_terms(q, cooc, norms, dev, min_count=mc, row_block=row_block,
                                             src=src).items():
                E[(norm, mc, src)] = arr
    return E


# ----------------------------------------------------------------------
#  Eq. (2) and (3) on a batch of dense queries
# ----------------------------------------------------------------------
@torch.no_grad()
def expand(Vq, E, alpha, alpha_mode="relative"):
    """V_q_ext = V_q + alpha_q * E.   Vq, E: (b, n_cols) dense tensors."""
    if alpha == 0:
        return Vq.clone()
    if alpha_mode == "absolute":
        return Vq + alpha * E
    qmax = Vq.max(dim=1).values
    emax = E.max(dim=1).values
    a = torch.where(emax > 0, alpha * qmax / emax.clamp_min(1e-30), torch.zeros_like(emax))
    return Vq + a[:, None] * E


@torch.no_grad()
def prune(Vext, Vq, spec, keep_original=True):
    """Eq. (3). See the module docstring for the specs."""
    orig = Vq > 0
    zero = torch.zeros((), device=Vext.device, dtype=Vext.dtype)
    if spec == "none":
        out = Vext
    else:
        kind, arg = spec.split(":")
        x = float(arg)
        if kind == "tau":
            out = torch.where(Vext > x, Vext, zero)
        elif kind == "rtau":
            thr = x * Vq.max(dim=1).values[:, None]
            out = torch.where(Vext > thr, Vext, zero)
        elif kind == "add":
            added = torch.where(orig, zero, Vext)
            kk = min(int(x), Vext.shape[1])
            vals, idx = torch.topk(added, kk, dim=1)
            out = torch.zeros_like(Vext).scatter_(1, idx, vals)
            out = torch.where(orig, Vext, out)
            keep_original = True
        else:
            raise ValueError(spec)
    if keep_original:
        out = torch.where(orig, Vext, out)
    return torch.where(out > 0, out, zero)


# ----------------------------------------------------------------------
#  exact search of several configurations in one pass over the query batches
# ----------------------------------------------------------------------
@torch.no_grad()
def search_configs(tdocs, q_csr, expansions, configs, k, dev, q_batch=256, doc_chunk=1_000_000, verbose=True):
    """For every LCEConfig: exact top-k over the documents (TorchCSR) of the expanded queries.
    expansions: {(norm, min_count, src): E numpy (nq x n_cols)}. Returns {config: dict(scores, rows, nnz, added)}."""
    nq = q_csr.n_rows
    k = min(k, tdocs.n_rows)
    res = {c: {"scores": np.empty((nq, k), np.float32), "rows": np.empty((nq, k), np.int64),
               "nnz": 0, "added": 0} for c in configs}
    n_batches = (nq + q_batch - 1) // q_batch
    for b, qs in enumerate(range(0, nq, q_batch)):
        qe = min(nq, qs + q_batch)
        Vq = csr_rows_to_dense(q_csr, np.arange(qs, qe), dev)
        Eb = {}
        for c in configs:
            if c.is_baseline:
                Vext = Vq
            else:
                key = c.e_key
                if key not in Eb:
                    Eb[key] = torch.from_numpy(expansions[key][qs:qe]).to(dev)
                Vext = prune(expand(Vq, Eb[key], c.alpha, c.alpha_mode), Vq, c.prune, c.keep_original)
            nz = Vext > 0
            res[c]["nnz"] += int(nz.sum())
            res[c]["added"] += int((nz & (Vq <= 0)).sum())
            s, i = search_with_query_table(tdocs, Vext.T.contiguous(), k, doc_chunk)
            res[c]["scores"][qs:qe] = s.float().cpu().numpy()
            res[c]["rows"][qs:qe] = i.cpu().numpy()
            del Vext
        del Vq, Eb
        if verbose and (b % max(1, n_batches // 10) == 0 or b == n_batches - 1):
            log(f"  search: {qe:,}/{nq:,} queries x {len(configs)} configurations")
    for c in configs:
        res[c]["avg_query_nnz"] = res[c]["nnz"] / max(1, nq)
        res[c]["avg_added"] = res[c]["added"] / max(1, nq)
    return res


# ----------------------------------------------------------------------
#  readable explanations
# ----------------------------------------------------------------------
def explain_row(vq, vext, n=10):
    """(original latents with their weight in V_q, added latents with their weight in V_q_ext), as
    'latent:weight' strings, strongest first."""
    vq = vq.detach().float().cpu()
    vext = vext.detach().float().cpu()
    orig_idx = torch.nonzero(vq > 0).flatten()
    o = orig_idx[torch.argsort(vq[orig_idx], descending=True)][:n].tolist()
    added = torch.where(vq > 0, torch.zeros_like(vext), vext)
    a_idx = torch.nonzero(added > 0).flatten()
    a = a_idx[torch.argsort(added[a_idx], descending=True)][:n].tolist()
    fmt = lambda ids, v: ", ".join(f"{i}:{v[i]:.2f}" for i in ids)  # noqa: E731
    return fmt(o, vq), fmt(a, vext)
