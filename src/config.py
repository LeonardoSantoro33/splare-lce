"""
config.py — Paths and settings shared by ALL scripts.

Set the two directories with environment variables (recommended, e.g. in ~/.bashrc):

    export SPLARE_DIR=/path/to/provided/splare/data      # pre-encoded SPLARE vectors (Seismic format), READ-ONLY
    export WORK_DIR=/path/to/large/scratch/splare_lce    # caches, co-occurrence matrix, splits, results

No path of any machine is written in the code: the defaults below are relative placeholders.
WORK_DIR needs ~90 GB: the co-occurrence matrix alone is ~61 GB (123.6k x 123.6k float32).

EVALUATION PROTOCOL (only MS MARCO dev small is available in the SPLARE space):
  the 6,980 dev queries are split ONCE, at random, with numpy.random.default_rng(42):
      val   1,000 queries   every choice of LCE (normalization of M, alpha, pruning) is made here
      test  5,980 queries   the configuration chosen on val is evaluated here, once
  The two sets are disjoint; nothing is ever chosen on test (03_split_dev.py, 06_validate_lce.py, 07_test_lce.py).
"""
import os
import re
from pathlib import Path

# Provided pre-encoded data (Seismic format). Never written to.
SPLARE_DIR = Path(os.environ.get("SPLARE_DIR", "data/splare"))

# Working directory (large).
WORK_DIR = Path(os.environ.get("WORK_DIR", "work"))

# Llama Scope 131k width (Llama-3.1-8B, residual stream of layer 26): 128 x 1024 SAE latents.
N_FEATURES = 131072

# File names in the provided directory.
PROVIDED_FILES = {
    "documents": "documents.bin",
    "queries": "queries.bin",
    "doc_ids": "doc_ids.npy",
    "query_ids": "queries_ids.npy",
    "mapping": "token_to_id_mapping.json",
    "groundtruth": "groundtruth_100.tsv",   # top-100 of each provided query (exact search)
    "groundtruth_10": "groundtruth.tsv",    # top-10
}

# Split of MS MARCO dev small into validation and test.
SPLIT_SEED = 42
N_VAL = 1000

# The three query sets. "dev" (all 6,980) is used ONLY for the un-expanded model: to check the data against the
# provided ground truth and to compare with the published SPLARE numbers. LCE is reported on "test" only.
QUERY_SETS = ("val", "test", "dev")


class Paths:
    """All paths derived from WORK_DIR, in one place."""

    def __init__(self, work_dir=None, splare_dir=None):
        self.work = Path(work_dir) if work_dir else WORK_DIR
        self.provided = Path(splare_dir) if splare_dir else SPLARE_DIR

        self.cache = self.work / "cache"                  # SPLARE vectors as CSR .npy (docs, dev queries)
        self.msmarco = self.work / "msmarco"              # MS MARCO dev: query texts and qrels
        self.splits = self.work / "splits"                # val / test qids
        self.cooc = self.work / "cooccurrence"            # the co-occurrence matrix M (Eq. 1)
        self.results = self.work / "results"
        self.runs = self.work / "runs"
        self.summary_csv = self.results / "summary.csv"

    # ---------------- provided files ----------------
    def provided_file(self, key):
        return self.provided / PROVIDED_FILES[key]

    # ---------------- caches ----------------
    @property
    def docs_csr(self):
        return self.cache / "docs"

    @property
    def dev_queries_csr(self):
        return self.cache / "queries_dev"

    # ---------------- MS MARCO dev ----------------
    @property
    def dev_queries_tsv(self):
        return self.msmarco / "queries.dev.small.tsv"

    @property
    def dev_qrels_tsv(self):
        return self.msmarco / "qrels.dev.small.tsv"

    # ---------------- split ----------------
    def split_file(self, name):
        return self.splits / f"{name}_qids.txt"

    @property
    def split_info(self):
        return self.splits / "split.json"

    # ---------------- results ----------------
    # A second, declared round of validation (06_validate_lce.py --tag NAME --after-test) writes the same files with
    # the tag in the name, so that the files of the first round are never overwritten.
    @staticmethod
    def _tagged(stem, ext, tag):
        if not tag:
            return f"{stem}.{ext}"
        if not re.fullmatch(r"[A-Za-z0-9_-]+", tag):
            raise SystemExit(f"[ERROR] --tag '{tag}': use only letters, digits, - and _")
        return f"{stem}_{tag}.{ext}"

    def best_config_for(self, tag=""):
        return self.results / self._tagged("lce_best", "json", tag)

    def validation_csv_for(self, tag=""):
        return self.results / self._tagged("lce_validation", "csv", tag)

    def best_per_norm_csv_for(self, tag=""):
        return self.results / self._tagged("lce_validation_best_per_norm", "csv", tag)

    @property
    def best_config(self):
        return self.best_config_for("")

    @property
    def validation_csv(self):
        return self.validation_csv_for("")

    def result_json(self, method, set_name):
        return self.results / f"{method}_{set_name}.json"

    def ensure_work_dir(self):
        """Create WORK_DIR (and subdirectories), with a clear message if that is not possible."""
        try:
            for d in [self.work, self.cache, self.msmarco, self.splits, self.cooc, self.results, self.runs]:
                d.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            raise SystemExit(
                f"\n[ERROR] Cannot create {self.work} (permission denied).\n"
                "Choose a directory you can write to and set it with:\n"
                "    export WORK_DIR=/path/to/scratch/$USER/splare_lce\n")
