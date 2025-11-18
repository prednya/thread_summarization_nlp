# `data/` Directory (for final packaging)

The project currently stores the normalized JSONL splits under:

- `data_splits/data/splits/`

with files like:

- `cs_sum_train.jsonl`, `cs_sum_dev.jsonl`, `cs_sum_test.jsonl`
- `dialogsum_train.jsonl`, `dialogsum_dev.jsonl`, `dialogsum_test.jsonl`
- `kaggle_train.jsonl`, `kaggle_dev.jsonl`, `kaggle_test.jsonl`
- `croco_train.jsonl`, `croco_dev.jsonl`, `croco_test.jsonl`

For Milestone 4, when you create the final tarball/zip, you can:

1. Move or copy these JSONL files into `data/`, **or**
2. Keep them in `data_splits/data/splits/` and update this README to explain
   the exact location and how to download/reconstruct them.

All code in `code/` looks for the data root via:

- Environment variable `THREAD_SUM_DATA_ROOT`, or
- Default path: `./data_splits/data/splits`

So if you place the files directly under `data/`, you can set:

```bash
export THREAD_SUM_DATA_ROOT=./data
```

before running any scripts.

