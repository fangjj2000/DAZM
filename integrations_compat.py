"""
Compatibility helpers for transformers integration APIs across versions.

Newer transformers releases remove some optional integrations such as SigOpt.
This module keeps the trainer imports stable by providing lightweight fallbacks
for symbols that may not exist in the installed version.
"""

from transformers.integrations import (
    get_reporting_integration_callbacks,
    hp_params,
    is_optuna_available,
    is_ray_tune_available,
    is_wandb_available,
    run_hp_search_optuna,
    run_hp_search_ray,
    run_hp_search_wandb,
)


def is_sigopt_available():
    return False


def run_hp_search_sigopt(*args, **kwargs):
    raise ImportError("SigOpt integration is not available in this transformers version.")

