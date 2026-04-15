# IMU datasets (local, user-provided)

This folder holds **empty layout only** (via `.gitkeep`) for **eight** wearable IMU HAR benchmarks (including **Opportunity** / OPPORTUNITY). **Download each dataset from the official source** and extract or copy files so paths match what `feeder/*_feeder.py` expects.

## Configure training

Point YAML `dataset.data_root` at this directory (absolute path recommended), e.g.:

```yaml
dataset:
  data_root: /path/to/Physics-Aware_Spiking_HAR/datasets
```

## Expected layout (relative to `data_root`)

| Dataset | Path | Notes |
|---------|------|--------|
| PAMAP2 | `pamap2+physical+activity+monitoring/PAMAP2_Dataset/PAMAP2_Dataset/<protocol>/subject*.dat` | `<protocol>` is e.g. `Protocol` per your YAML |
| Daily and Sports Activities | `daily+and+sports+activities/data/a##/p#/s##.txt` | UCI folder name uses `+` |
| TNDA-HAR | `TNDADATASET/*.csv` | Per-subject CSVs |
| HuGaDB | `HuGaDB/Data/*.txt` | Tab-separated; `relative_root` / `data_dir` overridable in YAML |
| USC-HAD | `USC-HAD/USC-HAD/Subject*/a*t*.mat` | Nested `USC-HAD` twice |
| HAR70+ | `har70/har70plus/*.csv` | |
| Parkinson (DAPHNet-style) | `Parkinson/dataset/S##R##.txt` | |
| Opportunity (UCI) | `OpportunityUCIDataset/dataset/S*.dat` | Whitespace-separated tables; optional `column_names.txt` in the same folder. YAML: `dataset.name: Opportunity`, `dataset.opportunity.relative_root: OpportunityUCIDataset/dataset`. You can symlink a full tree (e.g. `/path/to/your/OpportunityUCIDataset`) so that `data_root/OpportunityUCIDataset/dataset/` contains the UCI release. |

Full citations: see the main [README.md](../README.md) → **References & attributions**.

Do **not** commit large raw files to git; keep datasets only on disk or use your own storage policy.
