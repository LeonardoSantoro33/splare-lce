"""
data.py — Where every vector and every query set comes from.

  documents   the 8,841,823 MS MARCO passages, SPLARE vectors          cache/docs          (01_prepare_data.py)
  dev         the 6,980 MS MARCO dev small queries, SPLARE vectors     cache/queries_dev   (01_prepare_data.py)
  val / test  the random split of dev: 1,000 / 5,980 queries           splits/*_qids.txt   (03_split_dev.py)

Feature id = index of the SAE latent (0 .. 131,071): the keys of token_to_id_mapping.json, e.g. "60", "821".

THE SPLIT (03_split_dev.py). The dev qids that have both a vector and at least one relevant passage are sorted
in increasing order and permuted with numpy.random.default_rng(42); the first 1,000 are the validation set, the
others the test set. Both lists are then sorted again. The rule is deterministic: anyone with the same qids
gets the same split, and the two lists are saved (with their md5) so that the split never changes afterwards.
"""
import hashlib
from dataclasses import dataclass
from typing import Optional

import numpy as np

from config import N_VAL, QUERY_SETS, SPLIT_SEED
from msmarco import load_dev_qrels, load_dev_texts
from sparse_io import CSR
from utils import log, timer


@dataclass
class QuerySet:
    name: str
    q: CSR                       # query vectors (rows), ids = qids (int64), sorted
    qrels: dict                  # {qid: {pid: rel}}, only the qids of this set
    texts: Optional[dict]        # {qid: text} (None if queries.dev.small.tsv is missing)


# ----------------------------------------------------------------------
#  documents and dev queries
# ----------------------------------------------------------------------
def load_docs(paths, mmap=False):
    folder = paths.docs_csr
    if not (folder / "meta.json").exists():
        raise SystemExit(f"[ERROR] {folder} not found: run  python -u src/01_prepare_data.py")
    with timer("loading the SPLARE document vectors (cache)"):
        docs = CSR.load(folder, mmap=mmap)
    log(f"documents: {docs.n_rows:,}, non-zero entries: {docs.nnz:,} "
        f"(mean {docs.nnz / max(1, docs.n_rows):.1f} per document)")
    return docs


def load_dev_queries(paths):
    folder = paths.dev_queries_csr
    if not (folder / "meta.json").exists():
        raise SystemExit(f"[ERROR] {folder} not found: run  python -u src/01_prepare_data.py")
    q = CSR.load(folder)
    q.ids = np.asarray(q.ids).astype(np.int64)
    return q


# ----------------------------------------------------------------------
#  the validation / test split of dev
# ----------------------------------------------------------------------
def md5_of_ids(ids):
    return hashlib.md5("\n".join(str(int(x)) for x in ids).encode()).hexdigest()


def compute_split(dev_qids, qrels, n_val=N_VAL, seed=SPLIT_SEED):
    """(val qids, test qids, usable qids): disjoint, both sorted. Usable = has a vector and a relevant passage."""
    have_vec = set(np.asarray(dev_qids).astype(np.int64).tolist())
    usable = np.array(sorted(q for q, d in qrels.items() if q in have_vec and any(r > 0 for r in d.values())),
                      dtype=np.int64)
    if n_val >= len(usable):
        raise SystemExit(f"[ERROR] {n_val} validation queries requested, only {len(usable)} usable dev queries")
    perm = np.random.default_rng(seed).permutation(len(usable))
    val = np.sort(usable[perm[:n_val]])
    test = np.sort(usable[perm[n_val:]])
    return val, test, usable


def write_ids(path, ids):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{int(x)}\n" for x in ids))


def read_ids(path):
    return np.array([int(x) for x in path.read_text().split()], dtype=np.int64)


def load_split(paths, name):
    """qids of 'val' or 'test' (sorted), checked against the md5 saved by 03_split_dev.py."""
    from utils import load_json
    f = paths.split_file(name)
    if not f.exists() or not paths.split_info.exists():
        raise SystemExit(f"[ERROR] {f} not found: run  python -u src/03_split_dev.py")
    ids = read_ids(f)
    info = load_json(paths.split_info)
    if md5_of_ids(ids) != info[name]["md5"]:
        raise SystemExit(f"[ERROR] {f} was modified after 03_split_dev.py (md5 differs): the split must not change")
    return ids


# ----------------------------------------------------------------------
#  query sets
# ----------------------------------------------------------------------
def query_set(paths, name):
    """Vectors + qrels + texts of 'val', 'test' or 'dev' (all the dev queries with a relevant passage)."""
    if name not in QUERY_SETS:
        raise SystemExit(f"[ERROR] unknown query set {name!r}: use one of {QUERY_SETS}")
    q = load_dev_queries(paths)
    qrels_all = load_dev_qrels(paths)
    if name == "dev":
        wanted = np.array(sorted(x for x in q.ids.tolist()
                                 if x in qrels_all and any(r > 0 for r in qrels_all[x].values())), dtype=np.int64)
    else:
        wanted = load_split(paths, name)
    row_of = {int(x): i for i, x in enumerate(q.ids.tolist())}
    missing = [int(x) for x in wanted.tolist() if int(x) not in row_of]
    if missing:
        raise SystemExit(f"[ERROR] {len(missing)} qids of '{name}' have no vector (e.g. {missing[:5]})")
    rows = np.array([row_of[int(x)] for x in wanted.tolist()], dtype=np.int64)
    sub = q.subset(rows)
    sub.ids = wanted
    qrels = {int(x): qrels_all[int(x)] for x in wanted.tolist()}
    texts = load_dev_texts(paths)
    if texts is not None:
        texts = {int(x): texts.get(int(x), "") for x in wanted.tolist()}
    return QuerySet(name, sub, qrels, texts)
