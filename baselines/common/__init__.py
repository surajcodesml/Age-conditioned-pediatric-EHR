"""Common framework: interfaces, data contracts, metrics, counterfactual evaluation."""
from baselines.common.interface import BaselineModel, ModelOutput
from baselines.common.registry import REGISTRY, register_baseline, get_baseline
