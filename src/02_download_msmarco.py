"""
02_download_msmarco.py — The query texts and relevance judgments of MS MARCO dev small.

Extracts from the official collectionandqueries.tar.gz (~1 GB, downloaded and then deleted):
    queries.dev.small.tsv   the 6,980 dev queries (texts: only for the readable examples of 07)
    qrels.dev.small.tsv     the relevant passages of each dev query (every metric needs these)

If the two files are already somewhere on the machine, link them instead of downloading:
    python -u src/02_download_msmarco.py --link-from /path/to/folder/with/the/files
or use an already downloaded archive:
    python -u src/02_download_msmarco.py --from-tar /path/to/collectionandqueries.tar.gz
Network job, not a compute job: it can run on the login node.
"""
import argparse
import os
from pathlib import Path

from config import Paths
from msmarco import COLLECTION_AND_QUERIES, DEV_MEMBERS, download, extract_members, read_qrels, read_queries
from utils import log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-tar", default=None, help="use an already downloaded collectionandqueries.tar.gz")
    ap.add_argument("--link-from", default=None, help="folder that already contains the files: symlink them")
    ap.add_argument("--keep-tar", action="store_true", help="do not delete the .tar.gz after extraction")
    args = ap.parse_args()
    paths = Paths()
    paths.ensure_work_dir()
    out = paths.msmarco

    if args.link_from:
        src = Path(args.link_from)
        for name in DEV_MEMBERS:
            if (src / name).exists() and not (out / name).exists():
                os.symlink((src / name).resolve(), out / name)
                log(f"linked {name}")

    if all((out / m).exists() for m in DEV_MEMBERS):
        log("The dev files are already present.")
    else:
        if args.from_tar:
            tar_path = Path(args.from_tar)
        else:
            url, md5 = COLLECTION_AND_QUERIES
            tar_path = download(url, out / "collectionandqueries.tar.gz", md5)
        extract_members(tar_path, out, DEV_MEMBERS)
        if not args.keep_tar and not args.from_tar:
            tar_path.unlink()
            log("tar deleted (use --keep-tar to keep it)")

    queries = read_queries(paths.dev_queries_tsv)
    qrels = read_qrels(paths.dev_qrels_tsv)
    n_rel = sum(len(v) for v in qrels.values())
    log(f"  {paths.dev_queries_tsv.name:<24} {len(queries):>6,} queries")
    log(f"  {paths.dev_qrels_tsv.name:<24} {len(qrels):>6,} queries with judgments, {n_rel:,} relevant passages "
        f"({n_rel / max(1, len(qrels)):.2f} per query)")
    if len(qrels) != 6980:
        log("WARNING: MS MARCO dev small has 6,980 judged queries: is this the right file?")
    log("Done.")


if __name__ == "__main__":
    main()
