"""
Compatibility helpers for transformers utility APIs across versions.
"""

from transformers.utils import (
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_sagemaker_mp_enabled,
    logging,
)

try:
    from transformers.utils import is_torch_tpu_available
except ImportError:
    def is_torch_tpu_available(*args, **kwargs):
        return False


def trainer_get_learning_rate(trainer):
    if getattr(trainer, "lr_scheduler", None) is not None:
        try:
            last_lr = trainer.lr_scheduler.get_last_lr()
            if isinstance(last_lr, (list, tuple)) and len(last_lr) > 0:
                return last_lr[0]
        except Exception:
            pass

    if getattr(trainer, "optimizer", None) is not None and trainer.optimizer.param_groups:
        return trainer.optimizer.param_groups[0].get("lr", 0.0)

    return 0.0


def maybe_log_to_wandb(logs, step=None):
    try:
        import wandb
    except Exception:
        return

    if getattr(wandb, "run", None) is None:
        return

    try:
        if step is None:
            wandb.log(logs)
        else:
            current_step = getattr(getattr(wandb, "run", None), "step", None)
            if current_step is not None and step <= current_step:
                wandb.log(logs)
            else:
                wandb.log(logs, step=step)
    except Exception:
        return


def prefix_metric_keys(metrics, prefix):
    return {f"{prefix}{key}": value for key, value in metrics.items()}
