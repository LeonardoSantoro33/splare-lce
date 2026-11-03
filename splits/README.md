# Validation / test split of MS MARCO dev small

`src/03_split_dev.py` writes the split to `$WORK_DIR/splits/`. Copy the three files here once it is made, so that
the split is published with the code:

| File | Content |
|---|---|
| `val_qids.txt` | the 1,000 validation qids (one per line, sorted): every choice of LCE is made on these |
| `test_qids.txt` | the other 5,980 qids: the chosen configuration is evaluated on these, once |
| `split.json` | seed, sizes, md5 of each list, statistics of the two halves |

Rule: the dev qids with a SPLARE vector and at least one relevant passage are sorted in increasing order and
permuted with `numpy.random.default_rng(42)`; the first 1,000 are the validation set.
