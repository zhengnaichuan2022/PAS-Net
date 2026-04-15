"""
SNN and related model implementations.
"""
from .simple_snn_model import (
    SimpleSNNModel1D,
    SimpleSNNModel2D,
    SimpleSNNModel,
    create_simple_model
)

try:
    from .spikformer import (
        Spikeformer,
        SpikeformerWrapper,
        create_spikformer_model
    )
    __all__ = [
        'SimpleSNNModel1D',
        'SimpleSNNModel2D',
        'SimpleSNNModel',
        'create_simple_model',
        'Spikeformer',
        'SpikeformerWrapper',
        'create_spikformer_model',
    ]
except ImportError:
    __all__ = [
        'SimpleSNNModel1D',
        'SimpleSNNModel2D',
        'SimpleSNNModel',
        'create_simple_model',
    ]

try:
    from .spike_driven_transformer import (
        SpikeDrivenTransformer,
        SpikeDrivenTransformerWrapper,
        create_spike_driven_transformer_model
    )
    __all__.extend([
        'SpikeDrivenTransformer',
        'SpikeDrivenTransformerWrapper',
        'create_spike_driven_transformer_model',
    ])
except ImportError:
    pass

try:
    from .spike_driven_transformer_v2 import (
        SpikeDrivenTransformerV2,
        SpikeDrivenTransformerV2Wrapper,
        create_spike_driven_transformer_v2_model
    )
    __all__.extend([
        'SpikeDrivenTransformerV2',
        'SpikeDrivenTransformerV2Wrapper',
        'create_spike_driven_transformer_v2_model',
    ])
except ImportError:
    pass

try:
    from .stattn import (
        STAttnTransformer,
        STAttnTransformerWrapper,
        create_stattn_model
    )
    __all__.extend([
        'STAttnTransformer',
        'STAttnTransformerWrapper',
        'create_stattn_model',
    ])
except ImportError:
    pass

try:
    from .lmuformer import (
        LMUFormer,
        LMUFormerWrapper,
        create_lmuformer_model
    )
    __all__.extend([
        'LMUFormer',
        'LMUFormerWrapper',
        'create_lmuformer_model',
    ])
except ImportError:
    pass

try:
    from .seqsnn_har_models import (
        SpikeRNNHAR,
        SpikeGRUHAR,
        TSSNNHAR,
        SpikeTemporalConvNet2DHAR,
        create_spike_rnn_har_model,
        create_spike_gru_har_model,
        create_tssnn_har_model,
        create_spike_tcn2d_har_model,
    )
    __all__.extend([
        'SpikeRNNHAR',
        'SpikeGRUHAR',
        'TSSNNHAR',
        'SpikeTemporalConvNet2DHAR',
        'create_spike_rnn_har_model',
        'create_spike_gru_har_model',
        'create_tssnn_har_model',
        'create_spike_tcn2d_har_model',
    ])
except ImportError:
    pass

