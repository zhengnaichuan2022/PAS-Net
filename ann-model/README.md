# ANN Models (Subject-Independent Pipeline)

This folder contains ANN baselines adapted to the existing `train.py` pipeline.

Implemented models:
- `deep_conv_lstm_ann.py`
- `resnet_se_ann.py`
- `mch_cnn_gru_ann.py`
- `rtsfnet_ann.py` (PyTorch adaptation of rTsfNet core ideas: multi-head 3D rotation + TSF blocks)

All models accept input tensor shape `(B, T, C, V)` to stay compatible with current feeders and strict subject/session-independent split.

- `unihar_ann.py`
- `selfhar_ann.py`
- `if_convtransformer_ann.py`