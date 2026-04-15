"""
Factory: build the dataset feeder from config.
"""
from typing import Dict, Any
from .usc_had_feeder import USC_HAD_Feeder
from .daily_sports_feeder import DailySports_Feeder
from .pamap2_feeder import PAMAP2_Feeder
from .parkinson_feeder import Parkinson_Feeder
from .base_feeder import BaseFeeder


def create_feeder(config: Dict[str, Any], split: str = 'train') -> BaseFeeder:
    """
    Instantiate the feeder for ``config['dataset']['name']``.

    Args:
        config: Full training config dict
        split: One of ``train``, ``val``, ``test``

    Returns:
        A ``BaseFeeder`` subclass instance
    """
    dataset_name = config['dataset']['name']
    
    if dataset_name == 'USC-HAD':
        return USC_HAD_Feeder(config, split)
    elif dataset_name == 'Daily-Sports':
        return DailySports_Feeder(config, split)
    elif dataset_name == 'PAMAP2':
        return PAMAP2_Feeder(config, split)
    elif dataset_name == 'HAR70':
        from .har70_feeder import HAR70_Feeder
        return HAR70_Feeder(config, split)
    elif dataset_name == 'TNDA':
        from .tnda_feeder import TNDA_Feeder
        return TNDA_Feeder(config, split)
    elif dataset_name == 'Parkinson':
        return Parkinson_Feeder(config, split)
    elif dataset_name == 'HuGaDB':
        from .hugadb_feeder import HuGaDB_Feeder
        return HuGaDB_Feeder(config, split)
    elif dataset_name == 'Opportunity':
        from .opportunity_feeder import OpportunityFeeder
        return OpportunityFeeder(config, split)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

