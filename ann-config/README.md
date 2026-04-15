# ANN Configs

ANN experiment configs live here and are compatible with the same feeder/split and compute-energy proxy logging pipeline as SNN configs.

- Output root: `./logs-Subject-Independent-ANN`
- Split logic: `dataset.subject_independent_split: true`
- Runner: `scripts/run_ann_tmux_batches.sh`

Run one config:
```bash
python train.py --config ann-config/pamap2/deep_conv_lstm_ann.yaml
```

Run ANN batches (10 per batch by default):
```bash
scripts/run_ann_tmux_batches.sh
```

### HuGaDB

- New configs are in `ann-config/hugadb/*.yaml`.
- Data root is `dataset.data_root: ./datasets` (or another directory you pass in YAML) and `dataset.hugadb.relative_root: HuGaDB`.
- Subject-independent split uses the same `train_split/val_split/test_split` policy as other datasets.

