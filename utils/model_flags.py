"""Model / config predicates shared across train and tools."""
from pathlib import Path

_PAS_NET_MODEL_BASENAME = "PAS_Net.py"


def is_imuphysics_aware_spikeformer_config(config: dict) -> bool:
    """True iff the run uses the IMU Physics model registered in snn-model/PAS_Net.py."""
    m = config.get("model") or {}
    mtype = m.get("type", "")
    if mtype not in ("pas_net", "imu_physics_spikeformer", "imu_physics_aware_spikeformer"):
        return False
    mf = m.get("model_file") or ""
    try:
        name = Path(mf).name
    except Exception:
        name = str(mf)
    return name == _PAS_NET_MODEL_BASENAME
