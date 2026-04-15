"""
Model factory: instantiate networks from YAML ``model`` section.
"""
import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Dict, Any
import torch.nn as nn

def _import_module_from_model_file(model_file: str):
    """Load a module from a filesystem path when package import is not suitable."""
    file_path = Path(model_file)
    if not file_path.is_absolute():
        file_path = Path(__file__).parent.parent / file_path
    if not file_path.exists():
        return None
    module_name = f"dynamic_model_{file_path.stem}_{abs(hash(str(file_path)))}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[module_name] = module
    return module


def create_model(config: Dict[str, Any]) -> nn.Module:
    """
    Build a model from ``config['model']``.

    Args:
        config: Full training config dict

    Returns:
        ``nn.Module`` ready for ``.to(device)``.
    """
    model_config = config.get('model', {})
    model_type = model_config.get('type', 'simple_snn')
    model_file = model_config.get('model_file', 'snn-model/simple_snn_model.py')
    model_class = model_config.get('model_class', 'SimpleSNNModel')

    module_path = model_file.replace('.py', '').replace('/', '.')

    try:
        model_file_path = Path(model_file)
        if model_file_path.is_absolute():
            module = _import_module_from_model_file(model_file)
            if module is None:
                raise ImportError(f"Cannot import model from absolute path: {model_file}")
        elif module_path in sys.modules:
            module = sys.modules[module_path]
        else:
            try:
                module = importlib.import_module(module_path)
            except Exception:
                module = _import_module_from_model_file(model_file)
                if module is None:
                    raise

        create_func_name = None
        if model_type == 'spikformer':
            create_func_name = 'create_spikformer_model'
        elif model_type == 'spike_driven_transformer' or model_type == 'sdt':
            create_func_name = 'create_spike_driven_transformer_model'
        elif model_type == 'spike_driven_transformer_v2' or model_type == 'sdt_v2':
            create_func_name = 'create_spike_driven_transformer_v2_model'
        elif model_type == 'stattn' or model_type == 'st_atten':
            create_func_name = 'create_stattn_model'
        elif model_type == 'lmuformer' or model_type == 'lmu':
            create_func_name = 'create_lmuformer_model'
        elif model_type == 'imu_rhythm_neuromod_spikeformer' or model_type == 'imu_rhythm' or model_type == 'rhythm_neuromod':
            create_func_name = 'create_imu_rhythm_neuromod_spikeformer'
        elif model_type in (
            'pas_net',
            'imu_physics_spikeformer',
            'imu_physics_aware_spikeformer',
        ):
            create_func_name = (
                'create_pas_net'
                if hasattr(module, 'create_pas_net')
                else 'create_imu_physics_spikeformer'
            )
        elif model_type == 'qkformer_imu' or model_type == 'qkformer':
            create_func_name = 'create_qkformer_imu_model'
        elif model_type == 'spike_rnn_har' or model_type == 'spikernn':
            create_func_name = 'create_spike_rnn_har_model'
        elif model_type == 'spike_gru_har' or model_type == 'spikegru':
            create_func_name = 'create_spike_gru_har_model'
        elif model_type == 'tssnn_har' or model_type == 'tssnn':
            create_func_name = 'create_tssnn_har_model'
        elif model_type == 'spike_tcn2d_har' or model_type == 'spiketcn2d':
            create_func_name = 'create_spike_tcn2d_har_model'
        elif model_type == 'imu_s_expand_spikformer' or model_type == 'imu_spikformer_s':
            create_func_name = 'create_imu_s_expand_spikformer_model'
        elif model_type == 'deep_conv_lstm_ann':
            create_func_name = 'create_deep_conv_lstm_ann_model'
        elif model_type == 'resnet_se_ann':
            create_func_name = 'create_resnet_se_ann_model'
        elif model_type == 'mch_cnn_gru_ann':
            create_func_name = 'create_mch_cnn_gru_ann_model'
        elif model_type == 'rtsfnet_ann':
            create_func_name = 'create_rtsfnet_ann_model'
        elif model_type == 'unihar_ann':
            create_func_name = 'create_unihar_ann_model'
        elif model_type == 'selfhar_ann':
            create_func_name = 'create_selfhar_ann_model'
        elif model_type == 'if_convtransformer_ann':
            create_func_name = 'create_if_convtransformer_ann_model'
        elif model_type == 'simple_snn':
            create_func_name = 'create_simple_model'

        if create_func_name and hasattr(module, create_func_name):
            create_func = getattr(module, create_func_name)
            return create_func(config)

        if hasattr(module, 'create_simple_model'):
            return module.create_simple_model(config)

        if hasattr(module, 'create_spikformer_model'):
            return module.create_spikformer_model(config)

        if hasattr(module, 'create_spike_driven_transformer_model'):
            return module.create_spike_driven_transformer_model(config)

        if hasattr(module, 'create_spike_driven_transformer_v2_model'):
            return module.create_spike_driven_transformer_v2_model(config)

        if hasattr(module, 'create_stattn_model'):
            return module.create_stattn_model(config)

        if hasattr(module, 'create_lmuformer_model'):
            return module.create_lmuformer_model(config)

        if hasattr(module, 'create_imu_rhythm_neuromod_spikeformer'):
            return module.create_imu_rhythm_neuromod_spikeformer(config)

        if hasattr(module, 'create_pas_net'):
            return module.create_pas_net(config)
        if hasattr(module, 'create_imu_physics_spikeformer'):
            return module.create_imu_physics_spikeformer(config)

        if hasattr(module, 'create_spike_rnn_har_model'):
            return module.create_spike_rnn_har_model(config)
        if hasattr(module, 'create_spike_gru_har_model'):
            return module.create_spike_gru_har_model(config)
        if hasattr(module, 'create_tssnn_har_model'):
            return module.create_tssnn_har_model(config)
        if hasattr(module, 'create_spike_tcn2d_har_model'):
            return module.create_spike_tcn2d_har_model(config)
        if hasattr(module, 'create_imu_s_expand_spikformer_model'):
            return module.create_imu_s_expand_spikformer_model(config)

        if hasattr(module, model_class):
            ModelClass = getattr(module, model_class)

            num_classes = model_config.get('num_classes', 12)
            input_channels = model_config.get('input_channels', 3)

            if model_type == 'spikformer':
                embed_dims = model_config.get('embed_dims', 256)
                num_heads = model_config.get('num_heads', 16)
                mlp_ratio = model_config.get('mlp_ratio', 4.0)
                depth = model_config.get('depth', 2)
                drop_rate = model_config.get('drop_rate', 0.0)
                attn_drop_rate = model_config.get('attn_drop_rate', 0.0)
                drop_path_rate = model_config.get('drop_path_rate', 0.0)

                model = ModelClass(
                    in_channels=input_channels,
                    num_classes=num_classes,
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    depth=depth,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=drop_path_rate,
                )
                if hasattr(module, 'SpikeformerWrapper'):
                    return module.SpikeformerWrapper(model)
                else:
                    class SimpleWrapper(nn.Module):
                        def __init__(self, model):
                            super().__init__()
                            self.model = model
                        def forward(self, x):
                            x = x.permute(1, 0, 2, 3).contiguous()
                            return self.model(x)
                    return SimpleWrapper(model)
            else:
                num_imus = model_config.get('num_imus', 1)
                hidden_dim = model_config.get('hidden_dim', 128)
                time_steps = model_config.get('time_steps', 4)
                btcv_mode = model_config.get('btcv_mode', 1)

                model = ModelClass(
                    input_channels=input_channels,
                    num_imus=num_imus,
                    num_classes=num_classes,
                    hidden_dim=hidden_dim,
                    time_steps=time_steps,
                    mode=btcv_mode,
                )
                return model
        else:
            raise ValueError(f"Model class {model_class} or factory create_* not found in {module_path}")

    except ImportError as e:
        raise ImportError(f"Cannot import model module {module_path}: {e}")
    except AttributeError as e:
        raise AttributeError(f"Model class {model_class} missing: {e}")
    except Exception as e:
        raise RuntimeError(f"create_model failed: {e}")
