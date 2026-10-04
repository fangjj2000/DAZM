import contextlib
import functools
import glob
import inspect
import math
import os
import random
import re
import shutil
import sys
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import copy
import json
from metrics import f1
import numpy as np
from tqdm.auto import tqdm
from transformers import Trainer
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV

# Integrations must be imported before ML frameworks:
from integrations_compat import (
    get_reporting_integration_callbacks,
    hp_params,
    is_optuna_available,
    is_ray_tune_available,
    is_sigopt_available,
    is_wandb_available,
    run_hp_search_optuna,
    run_hp_search_ray,
    run_hp_search_sigopt,
    run_hp_search_wandb,
)

import numpy as np
import torch
import torch.distributed as dist
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import __version__
from transformers.configuration_utils import PretrainedConfig
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.dependency_versions_check import dep_version_check
from transformers.modelcard import TrainingSummary
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_MAPPING_NAMES
from transformers.optimization import Adafactor, get_scheduler
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    IterableDatasetShard,
    LengthGroupedSampler,
    nested_concat,
    reissue_pt_warnings,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    FSDPOption,
    HPSearchBackend,
    HubStrategy,
    IntervalStrategy,
    PredictionOutput,
    RemoveColumnsCollator,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    denumpify_detensorize,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    has_length,
    number_of_arguments,
    seed_worker,
    set_seed,
    speed_metrics,
)
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from utils_compat import (
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
    logging,
    trainer_get_learning_rate,
    maybe_log_to_wandb,
    prefix_metric_keys,
)
from transformers.utils.generic import ContextManagers
from lr_scheduler import zo_lr_scheduler
from Hessian_smooth_scheduler import Hessian_smooth_scheduler

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from .utils import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

if is_apex_available():
    try:
        from apex import amp
    except ImportError:
        amp = None

if is_datasets_available():
    import datasets

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from .trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

if TYPE_CHECKING:
    import optuna

logger = logging.get_logger(__name__)
# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"


def zeropower_via_newtonschulz5(G):
    assert G.ndim >= 2 
    a, b, c = (3.4445, -4.7750,  2.0315)
    #X = G.bfloat16()
    X=G
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(5):
        A = X @ X.mT
        B = b * A + c * A @ A 
        X = a * X + B @ X
    
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

def zeropower_via_svd(G):
    """
    Orthogonalize G directly using SVD instead of Newton–Schulz iteration.
    Works on float16, bfloat16, float32, or float64.
    For half-precision types, we upcast to float32 before doing SVD.
    """
    assert G.ndim >= 2, "Input must be at least 2D (batched matrices allowed)"

    # make sure smaller dimension is last for SVD efficiency
    transpose_needed = G.size(-2) > G.size(-1)
    if transpose_needed:
        G = G.mT

    orig_dtype = G.dtype
    if G.dtype in (torch.float16, torch.bfloat16):
        G32 = G.float()  # upcast for numerical stability and CUDA kernel support
        U, S, Vh = torch.linalg.svd(G32, full_matrices=False)
        X = (U @ Vh).to(orig_dtype)
    else:
        U, S, Vh = torch.linalg.svd(G, full_matrices=False)
        X = U @ Vh

    if transpose_needed:
        X = X.mT
    return X


import torch

class GreedyZuonTrainer(Trainer):

    def _get_learning_rate(self):
        return trainer_get_learning_rate(self)
    
    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        """
        We overload the original training loop to add linear probing and MeZO. Search key word "MeZO added"
        for those updates.
        """
        self._train_batch_size = batch_size
        self.do_grad_scaling = self.args.do_grad_scaling
        self.best_eval_loss = 100.0
        # Keep an independent stop flag.  TrainerControl can be replaced by
        # callback handlers during evaluation; the flag must survive that
        # round-trip so overfit early stopping cannot accidentally continue
        # spending steps (and memory) after the stop decision.
        self._stop_requested = False
        train_dataloader = self.get_train_dataloader()
        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size #16*1*1

        len_dataloader = None
        if has_length(train_dataloader): 
            len_dataloader = len(train_dataloader) 
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader) 
            if args.max_steps > 0: 
                max_steps = args.max_steps #20000
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                ) #318
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size 
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torch.distributed.launch)."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = False
        self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing: 
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        if is_sagemaker_mp_enabled() and resume_from_checkpoint is not None: 
            self._load_from_checkpoint(resume_from_checkpoint, model)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model: 
            self.model_wrapped = model

        if delay_optimizer_creation: 
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)
        self._load_optimizer_and_scheduler(resume_from_checkpoint)
        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")
        logger.info(
            f"  Number of trainable parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ): 
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "
                    "flag to your launch command, but you will resume the training on data already seen by your model."
                )
                if self.is_local_process_zero() and not args.disable_tqdm:
                    steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description("Skipping the first batches")

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None: 
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else: 
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip: 
            for epoch in range(epochs_trained):
                is_random_sampler = hasattr(train_dataloader, "sampler") and isinstance(
                    train_dataloader.sampler, RandomSampler
                )
                if  not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    # That was before PyTorch 1.11 however...
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    _ = list(train_dataloader.sampler)

        
        for epoch in range(epochs_trained, num_train_epochs):
            

            zo_learning_rate = zo_lr_scheduler(self.args.learning_rate, self.args.zo_lr_scheduler_type, self.args.warmup_step, self.args.decay_step, self.state.global_step, int(num_train_epochs))
            Hessian_smooth = Hessian_smooth_scheduler(self.args.hessian_smooth_type, self.state.global_step, int(num_train_epochs))
            
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [args.device]).per_device_loader(args.device)
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            step = -1
            self.named_parameters_to_optim = []
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.named_parameters_to_optim.append((name, param))
            for step, inputs in enumerate(epoch_iterator):
                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                tr_loss_step = self.lowdim_zo_step(model, inputs)

                if (
                    args.logging_nan_inf_filter
                    and not is_torch_tpu_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:  
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                self.lowdim_zo_update()

                self.state.global_step += 1
                self.state.epoch = epoch + (step + 1) / steps_in_epoch
                self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

                if (
                    self.control.should_epoch_stop
                    or self.control.should_training_stop
                    or getattr(self, "_stop_requested", False)
                ):
                    break

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop or getattr(self, "_stop_requested", False):
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")

        run_dir = self._get_output_dir(trial)
        best_dir = os.path.join(run_dir, "checkpoint-best")
        last_step = self.state.global_step
        best_step = getattr(self, "best_checkpoint_step", None)
        last_is_best = (
            self.state.best_model_checkpoint is not None
            and best_step is not None
            and int(best_step) == int(last_step)
        )

        # Always have checkpoint-best.
        if self.state.best_model_checkpoint is None:
            logger.warning("No validation-best checkpoint yet; saving final model as checkpoint-best.")
            os.makedirs(best_dir, exist_ok=True)
            self.save_model(best_dir, _internal_call=True)
            self.state.best_model_checkpoint = best_dir
            self.best_checkpoint_step = last_step
            last_is_best = True
            if self.args.should_save:
                self.state.save_to_json(os.path.join(best_dir, TRAINER_STATE_NAME))
        else:
            logger.info(
                f"Best model checkpoint [{self.state.best_model_checkpoint}] "
                f"(step={best_step}, metric={self.state.best_metric})"
            )

        # If last != best, also keep the final model as checkpoint-{last_step}.
        # If last == best, only checkpoint-best is retained.
        if last_is_best:
            self.last_model_checkpoint = self.state.best_model_checkpoint
            logger.info(
                f"Last step {last_step} is also best; retaining only checkpoint-best."
            )
        else:
            last_dir = os.path.join(run_dir, f"{PREFIX_CHECKPOINT_DIR}-{last_step}")
            os.makedirs(last_dir, exist_ok=True)
            self.save_model(last_dir, _internal_call=True)
            self.last_model_checkpoint = last_dir
            if self.args.should_save:
                self.state.save_to_json(os.path.join(last_dir, TRAINER_STATE_NAME))
            logger.info(
                f"Last step {last_step} differs from best step {best_step}; "
                f"kept both checkpoint-best and {last_dir}."
            )

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        if torch.cuda.is_available():
            self.gpu_memory_max_allocated_mb = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)
            metrics["gpu_memory_max_allocated_mb"] = self.gpu_memory_max_allocated_mb
        else:
            self.gpu_memory_max_allocated_mb = None

        self.log(metrics)
        maybe_log_to_wandb(metrics, step=self.state.global_step)

        # Remove any other leftover periodic checkpoints.
        keep = {
            os.path.abspath(self.state.best_model_checkpoint),
            os.path.abspath(self.last_model_checkpoint),
        }
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir) if hasattr(self, "_sorted_checkpoints") else []
        for checkpoint in checkpoints_sorted:
            if os.path.abspath(checkpoint) not in keep:
                logger.info(f"Deleting leftover checkpoint [{checkpoint}]")
                shutil.rmtree(checkpoint, ignore_errors=True)
        logger.info(
            f"Retained checkpoints: best={self.state.best_model_checkpoint}"
            + ("" if last_is_best else f", last={self.last_model_checkpoint}")
        )
        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        
        return TrainOutput(self.state.global_step, train_loss, metrics)
    

    def _sample_orthogonal_u(self, m, r, device, dtype, random_seed):
        A = torch.randn(m, r, device=device, dtype=torch.float32)
        Q, R = torch.linalg.qr(A, mode="reduced")

        # Randomize signs to make it Haar distributed
        signs = torch.sign(torch.diagonal(R))
        # If any diag entries are 0 (extremely rare), set to +1
        signs[signs == 0] = 1.0
        Q = Q * signs

        # Optional: downcast
        if dtype in (torch.float16, torch.bfloat16):
            return Q.to(dtype=dtype)
        elif dtype in (torch.float32, torch.float64):
            return Q.to(dtype)
        else:
            raise TypeError(f"Unsupported dtype for P matrix: {dtype}")
        
    def _sample_orthogonal_uv(self, m, n, r, device, dtype, random_seed,):
        if r > min(m, n):
            raise ValueError(
                f"rank r={r} must satisfy r <= min(m, n)={min(m, n)}."
            )

        generator = torch.Generator(device=device)
        generator.manual_seed(random_seed)

        # Generate the left orthogonal basis U
        A_u = torch.randn(m, r, device=device, dtype=torch.float32, generator=generator,)
        U, R_u = torch.linalg.qr(A_u, mode="reduced")

        # Randomize column signs to obtain a Haar-distributed basis
        signs_u = torch.sign(torch.diagonal(R_u))
        signs_u[signs_u == 0] = 1.0
        U = U * signs_u

        # Generate the right orthogonal basis V
        A_v = torch.randn(n, r, device=device, dtype=torch.float32, generator=generator,)
        V, R_v = torch.linalg.qr(A_v, mode="reduced")

        # Randomize column signs to obtain a Haar-distributed basis
        signs_v = torch.sign(torch.diagonal(R_v))
        signs_v[signs_v == 0] = 1.0
        V = V * signs_v

        if dtype in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        ):
            return U.to(dtype=dtype), V.to(dtype=dtype)

        raise TypeError(f"Unsupported dtype for U and V matrices: {dtype}")

    def _replace_uv_subspace(self, name, new_u, new_v):
        """Replace a parameter's bases while preserving full-space momentum."""
        old_momentum = self.momentum.get(name)
        if old_momentum is not None:
            old_u = self.u_matrices[name]
            old_v = self.v_matrices[name]
            left_projection = new_u.float().T @ old_u.float()
            right_projection = old_v.float() @ new_v.float().T
            projected_momentum = (
                left_projection @ old_momentum.float() @ right_projection
            )
            self.momentum[name] = projected_momentum.to(
                device=old_momentum.device,
                dtype=old_momentum.dtype,
            )

        self.u_matrices[name] = new_u
        self.v_matrices[name] = new_v

    def _random_orthogonal_uv(self, random_seed):
        """Replace every matrix parameter's U/V bases with random bases."""
        for param_idx, (name, param) in enumerate(self.named_parameters_to_optim):
            if param.data.ndim < 2:
                continue
            m, n = param.data.shape
            u, v = self._sample_orthogonal_uv(
                m,
                n,
                self.args.rank_r,
                param.data.device,
                param.data.dtype,
                random_seed + param_idx,
            )
            self._replace_uv_subspace(name, u, v.T)

    def _greedy_orthogonal_u(self):
        args = self.args
        rank_r = args.rank_r
        energy_threshold = args.energy_threshold

        for name, param in self.named_parameters_to_optim:
            if param.data.ndim >= 2:

                # exploit
                u_exploit, S_v, _ = torch.svd_lowrank(self.u_matrices[name].float() @ self.period_z[name].float(), q=rank_r,  niter=4)
                p_keep_min = max(1, rank_r // 4)
                p_keep_max = max(1, int(0.75 * rank_r))
                energy = torch.cumsum(S_v ** 2, dim=0) / (torch.sum(S_v ** 2) + 1e-12)
                threshold = torch.tensor(energy_threshold, device=energy.device, dtype=energy.dtype)
                p_keep_adapt = int(torch.searchsorted(energy, threshold).item() + 1)
                # print(p_keep_adapt)
                p_keep_adapt = min(max(p_keep_adapt, p_keep_min), p_keep_max)
                u_keep = u_exploit[:, :p_keep_adapt]

                # explore
                w = torch.randn(param.data.shape[0], rank_r - p_keep_adapt, device=param.data.device, dtype=torch.float32)
                explore_matrix = w - u_keep @ (u_keep.T @ w)
                scale = math.sqrt((param.data.shape[0]-p_keep_adapt)/(rank_r - p_keep_adapt))
                u_explore, _ = torch.linalg.qr(explore_matrix, mode="reduced")
                u_total = torch.cat([u_keep, scale * u_explore], dim=1)
                u_total, _ = torch.linalg.qr(u_total, mode="reduced")
                self.u_matrices[name] = u_total.to(param.device).to(param.dtype)

    def _greedy_orthogonal_uv(self):
        args = self.args
        rank_r = args.rank_r
        p = args.p
        energy_threshold = args.energy_threshold
        self._last_refresh_metadata = {}

        for name, param in self.named_parameters_to_optim:
            if param.data.ndim >= 2:

                # exploit
                u_exploit, S_v, v_exploit = torch.linalg.svd(
                    self.period_z[name].float(),
                    full_matrices=False
                )

                p_keep_min = max(1, rank_r // 4)
                p_keep_max = max(1, int(0.75 * rank_r))

                energy = torch.cumsum(
                    S_v ** 2,
                    dim=0,
                ) / (torch.sum(S_v ** 2))

                threshold = torch.tensor(
                    energy_threshold,
                    device=energy.device,
                    dtype=energy.dtype,
                )

                # searchsorted can return len(S_v); +1 then exceeds rank_r → negative explore dim.
                p_keep_adapt = int(
                    torch.searchsorted(
                        energy,
                        threshold,
                    ).item() + 1
                )
                p_keep_adapt = min(max(p_keep_adapt, p_keep_min), p_keep_max)
                p_keep_adapt = min(p_keep_adapt, int(S_v.numel()), rank_r - 1)
                p_keep_adapt = max(1, p_keep_adapt)
                explore_dim = rank_r - p_keep_adapt

                # Keep paired left and right singular vectors
                u_exploit = self.u_matrices[name].float() @ u_exploit
                v_exploit = v_exploit @ self.v_matrices[name].float()
                u_keep = u_exploit[:, :p_keep_adapt]
                v_keep = v_exploit[:p_keep_adapt, :]
                exploit_u = u_keep.clone()
                exploit_r = v_keep.T.clone()
                # print(f"Keeping {p_keep_adapt} singular vectors for parameter {name}")

                if explore_dim > 0:
                    # ==========================================================
                    # Explore left subspace
                    # ==========================================================
                    w_u = torch.randn(
                        param.data.shape[0],
                        explore_dim,
                        device=param.data.device,
                        dtype=torch.float32,
                    )

                    u_explore_matrix = w_u - u_keep @ (
                        u_keep.T @ w_u
                    )

                    u_explore, _ = torch.linalg.qr(
                        u_explore_matrix,
                        mode="reduced",
                    )

                    u_total = torch.cat(
                        [
                            u_keep,
                            u_explore[:, :explore_dim],
                        ],
                        dim=1,
                    )

                    # ==========================================================
                    # Explore right subspace
                    # ==========================================================
                    w_v = torch.randn(
                        param.data.shape[1],
                        explore_dim,
                        device=param.data.device,
                        dtype=torch.float32,
                    )

                    v_explore_matrix = w_v - v_keep.T @ (v_keep @ w_v)

                    v_explore, _ = torch.linalg.qr(
                        v_explore_matrix,
                        mode="reduced",
                    )

                    v_total = torch.cat(
                        [
                            v_keep,
                            v_explore[:, :explore_dim].T,
                        ],
                        dim=0,
                    )
                else:
                    u_total = u_keep
                    v_total = v_keep

                # Ensure exact rank_r columns/rows after QR truncation edge cases.
                if u_total.shape[1] < rank_r:
                    pad = rank_r - u_total.shape[1]
                    w_u = torch.randn(
                        param.data.shape[0], pad, device=param.data.device, dtype=torch.float32
                    )
                    w_u = w_u - u_total @ (u_total.T @ w_u)
                    u_pad, _ = torch.linalg.qr(w_u, mode="reduced")
                    u_total = torch.cat([u_total, u_pad[:, :pad]], dim=1)
                if v_total.shape[0] < rank_r:
                    pad = rank_r - v_total.shape[0]
                    w_v = torch.randn(
                        param.data.shape[1], pad, device=param.data.device, dtype=torch.float32
                    )
                    w_v = w_v - v_total.T @ (v_total @ w_v)
                    v_pad, _ = torch.linalg.qr(w_v, mode="reduced")
                    v_total = torch.cat([v_total, v_pad[:, :pad].T], dim=0)

                u_total = u_total[:, :rank_r]
                v_total = v_total[:rank_r, :]

                new_u = u_total.to(device=param.device, dtype=param.dtype)
                new_v = v_total.to(device=param.device, dtype=param.dtype)

                self._replace_uv_subspace(name, new_u, new_v)

                if getattr(args, "capture_displacement", False):
                    self._last_refresh_metadata[name] = {
                        "p": int(p_keep_adapt),
                        "rank": int(rank_r),
                        "u_exploit": exploit_u,
                        "r_exploit": exploit_r,
                        "u_new": new_u,
                        "v_new": new_v,
                    }

    def _capture_layer_names(self):
        configured = [x.strip() for x in getattr(self.args, "capture_layers", "").split(",") if x.strip()]
        if configured:
            available = [name for name, param in self.named_parameters_to_optim if param.data.ndim == 2]
            selected = []
            for token in configured:
                for name in available:
                    if token in name and name not in selected:
                        selected.append(name)
            return selected
        # Keep the default small and model-agnostic: first and middle attention q/v projections.
        names = [name for name, param in self.named_parameters_to_optim
                 if param.data.ndim == 2 and ("q_proj.weight" in name or "v_proj.weight" in name)]
        return names[:2]

    @staticmethod
    def _sin_theta_operator(left, target):
        """Largest principal sine between two column-orthonormal bases."""
        left = left.float()
        target = target.float()
        singular_values = torch.linalg.svdvals(left.T @ target)
        min_cosine = singular_values.min().clamp(0.0, 1.0)
        return torch.sqrt((1.0 - min_cosine.square()).clamp_min(0.0)).item()

    def _capture_displacement_metrics(self, model, inputs):
        """Measure rho >= gamma - epsilon^2 on selected layers at a refresh.

        The gradient is exact for the current probe batch.  Results are appended
        as JSONL so an interrupted supplementary run remains analyzable.
        """
        args = self.args
        if not getattr(args, "capture_displacement", False):
            return
        refresh_step = int(getattr(self, "step", 0))
        stride = max(1, int(getattr(args, "capture_refresh_stride", 5)))
        refresh_id = refresh_step // max(1, int(args.step_interval))
        names_to_measure = set(self._capture_layer_names())
        metadata = getattr(self, "_last_refresh_metadata", {})
        selected = [name for name in names_to_measure if name in metadata]
        if not selected:
            return

        output_file = getattr(args, "capture_output_file", None)
        if not output_file:
            output_file = os.path.join(args.output_dir, "displacement_capture.jsonl")
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

        # Keep p for every refresh even when the expensive exact-gradient probe
        # is sampled sparsely.
        with open(output_file, "a", encoding="utf-8") as handle:
            for name in selected:
                handle.write(json.dumps({
                    "record_type": "p",
                    "refresh_step": refresh_step,
                    "refresh_id": refresh_id,
                    "layer": name,
                    "rank": int(metadata[name]["rank"]),
                    "p": int(metadata[name]["p"]),
                }, sort_keys=True) + "\n")

        if refresh_id % stride != 0:
            return

        parameter_map = dict(self.named_parameters_to_optim)
        selected_params = [parameter_map[name] for name in selected if name in parameter_map]
        if not selected_params:
            return

        was_training = model.training
        model.eval()
        prepared_inputs = self._prepare_inputs(inputs)
        model.zero_grad(set_to_none=True)
        start_time = time.time()
        try:
            with torch.enable_grad():
                loss = self.compute_loss(model, prepared_inputs)
                if isinstance(loss, tuple):
                    loss = loss[0]
                gradients = torch.autograd.grad(
                    loss,
                    selected_params,
                    allow_unused=True,
                    retain_graph=False,
                    create_graph=False,
                )

            rows = []
            for name, gradient in zip(selected, gradients):
                if gradient is None or gradient.ndim != 2 or name not in metadata:
                    continue
                h = gradient.detach().float()
                if not torch.isfinite(h).all() or h.numel() == 0:
                    continue
                u_svd, singular_values, v_h = torch.linalg.svd(h, full_matrices=False)
                layer_meta = metadata[name]
                p_keep = min(int(layer_meta["p"]), int(singular_values.numel()))
                if p_keep <= 0:
                    continue
                leading_u = u_svd[:, :p_keep]
                leading_v = v_h[:p_keep, :].T
                nuclear_norm = singular_values.sum().item()
                gamma = singular_values[:p_keep].sum().item() / max(nuclear_norm, 1e-12)
                eps_u = self._sin_theta_operator(layer_meta["u_exploit"][:, :p_keep], leading_u)
                eps_v = self._sin_theta_operator(layer_meta["r_exploit"][:, :p_keep], leading_v)
                epsilon = max(eps_u, eps_v)
                projected = layer_meta["u_new"].float().T @ h @ layer_meta["v_new"].float().T
                rho = torch.linalg.svdvals(projected).sum().item() / max(nuclear_norm, 1e-12)
                bound = gamma - epsilon * epsilon
                rows.append({
                    "record_type": "metric",
                    "refresh_step": refresh_step,
                    "refresh_id": refresh_id,
                    "layer": name,
                    "rows": int(h.shape[0]),
                    "cols": int(h.shape[1]),
                    "rank": int(layer_meta["rank"]),
                    "p": p_keep,
                    "gamma": gamma,
                    "epsilon_u": eps_u,
                    "epsilon_v": eps_v,
                    "epsilon_exploit": epsilon,
                    "epsilon_squared": epsilon * epsilon,
                    "rho": rho,
                    "bound": bound,
                    "margin": rho - bound,
                    "bound_holds": bool(rho + 1e-6 >= bound),
                    "probe_loss": float(loss.detach().item()),
                    "measurement_seconds": time.time() - start_time,
                })
            if rows:
                with open(output_file, "a", encoding="utf-8") as handle:
                    for row in rows:
                        handle.write(json.dumps(row, sort_keys=True) + "\n")
        finally:
            model.zero_grad(set_to_none=True)
            if was_training:
                model.train()

    # def _greedy_orthogonal_u(self, model, inputs, base_seed):
    #     self.fulldim_zo_step(model, inputs, base_seed)
    #     self.fulldim_zo_update_u(base_seed)


    def lowdim_zo_perturb_parameters(self, random_seed=None, scaling_factor=1, sample_idx=0):
        """
        Perform parameter perturbation for low-dimensional ZO gradient estimation.
        """
        args = self.args
        # Set seed to ensure u is consistent between the (+) and (-) perturbation of the same sample
        torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)
        rank_r = args.rank_r
        
        for name, param in self.named_parameters_to_optim:
            if name not in self.z:
                self.z[name] = {}
            if name not in self.u_matrices:
                self.u_matrices[name] = {}
            if param.data.ndim >= 2:
                m, n = param.data.shape
                # Standard Gaussian sampling
                z = torch.randn(rank_r, rank_r, device=param.data.device, dtype=param.data.dtype)

                self.z[name][sample_idx] = z
                u = self.u_matrices[name]
                v = self.v_matrices[name]
                # Perturb parameters: W' = W + u @ v * scaling
                perturb = (u @ z @ v) * (scaling_factor * args.zo_eps)
                param.data.add_(perturb)

            else:
                # 1D case (no subspace)
                z = torch.randn_like(param.data)
                self.z[name][sample_idx] = z 
                param.data.add_(z * (scaling_factor * args.zo_eps))

    def fulldim_zo_perturb_parameters(self, random_seed=None, scaling_factor=1, sample_index=0):
        """
        Perform parameter perturbation for full-dimensional ZO gradient estimation.
        """
        args = self.args
        # Set seed to ensure u is consistent between the (+) and (-) perturbation of the same sample
        
        for param_idx, (name, param) in enumerate(self.named_parameters_to_optim):
            if param.data.ndim < 2:
                continue
            torch.manual_seed(random_seed + 1000000*sample_index + param_idx)
            z = torch.randn_like(param.data)
            param.data.add_(z * (scaling_factor * args.zo_eps))


    def zo_forward(self, model, inputs):
        """
        Get (no gradient) loss from the model. Dropout is turned off too.
        """
        model.eval()
        with torch.inference_mode():
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                # Warning: this is copied from the original Huggingface Trainer. Untested.
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
        return loss.detach()

    def lowdim_zo_step(self, model, inputs):
        """
        Low-dimensional random gradient estimate step.
        """
        args = self.args
        base_seed = np.random.randint(1000000000)
        if hasattr(self, 'step'):
            self.step += 1
        else:
            self.subspace_update_mode = getattr(
                args, "subspace_update_mode", "gradient"
            ).lower()
            valid_subspace_modes = {"gradient", "frozen", "random"}
            if self.subspace_update_mode not in valid_subspace_modes:
                raise ValueError(
                    "subspace_update_mode must be one of "
                    f"{sorted(valid_subspace_modes)}, got "
                    f"{self.subspace_update_mode!r}."
                )
            logger.info(
                "Subspace update mode: %s (interval=%s)",
                self.subspace_update_mode,
                args.step_interval,
            )
            self.step = 0
            self.u_matrices = {}
            self.v_matrices = {}
            self.momentum = {}
            self.period_z = {}
            self.z = {}
            for name, param in self.named_parameters_to_optim:
                if param.data.ndim >= 2:
                    m, n = param.data.shape
                    u, v = self._sample_orthogonal_uv(m, n, args.rank_r, param.data.device, param.data.dtype, base_seed)
                    self.u_matrices[name] = u
                    self.v_matrices[name] = v.T
                    
        resample = (self.step != 0) and (self.step % args.step_interval == 0)
        if resample:
            if self.subspace_update_mode == "gradient":
                self._greedy_orthogonal_uv()
                self._capture_displacement_metrics(model, inputs)
                self.period_z = {}
            elif self.subspace_update_mode == "random":
                self._random_orthogonal_uv(base_seed)

        self.projected_grads_list = []
        self.z = {} 

        if args.zo_perturbation_mode == 'one_side':
            num_samples = args.num_samples
            loss_baseline = self.zo_forward(model, inputs)
            
            for i in range(num_samples):
                current_seed = base_seed + i
                self.lowdim_zo_perturb_parameters(random_seed=current_seed, scaling_factor=1, sample_idx=i)
                loss_perturbed = self.zo_forward(model, inputs)
                grad_est = ((loss_perturbed - loss_baseline) / self.args.zo_eps).item()
                self.projected_grads_list.append(grad_est)
                self.lowdim_zo_perturb_parameters(random_seed=current_seed, scaling_factor=-1, sample_idx=i)

            return loss_baseline

        else:
            num_samples = args.num_samples
            for i in range(num_samples):
                current_seed = base_seed + i
                self.lowdim_zo_perturb_parameters(random_seed=current_seed, scaling_factor=1, sample_idx=i)
                loss1 = self.zo_forward(model, inputs)

                self.lowdim_zo_perturb_parameters(random_seed=current_seed, scaling_factor=-2, sample_idx=i)
                loss2 = self.zo_forward(model, inputs)

                grad_est = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()
                self.projected_grads_list.append(grad_est)

                self.lowdim_zo_perturb_parameters(random_seed=current_seed, scaling_factor=1, sample_idx=i)

            return (loss1+loss2)/2

    def fulldim_zo_step(self, model, inputs, base_seed):
        """
        full-dimensional random gradient estimate step.
        """
        args = self.args
        self.projected_grads_list_MeZO = []
  
        if getattr(args, 'num_samples_MeZO', False) :
            num_samples_MeZO = args.num_samples_MeZO
            for i in range(num_samples_MeZO):
                self.fulldim_zo_perturb_parameters(random_seed=base_seed, scaling_factor=1, sample_index=i)
                loss1 = self.zo_forward(model, inputs)
                self.fulldim_zo_perturb_parameters(random_seed=base_seed, scaling_factor=-2, sample_index=i)
                loss2 = self.zo_forward(model, inputs)
                grad_est = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()
                self.projected_grads_list_MeZO.append(grad_est)
                self.fulldim_zo_perturb_parameters(random_seed=base_seed, scaling_factor=1, sample_index=i)


    def lowdim_zo_update(self):
        args = self.args
        lr = self._get_learning_rate()
        rank_r = args.rank_r
        opt_type = args.zo_optimizer.lower()

        num_samples = len(self.projected_grads_list)

        for name, param in self.named_parameters_to_optim:
            if param.data.ndim >= 2:
                u = self.u_matrices[name]
                v = self.v_matrices[name]
                lowdim_rge = torch.zeros(rank_r, rank_r, device=param.data.device, dtype=param.data.dtype)
                
                for i in range(num_samples):
                    z_i = self.z[name][i] 
                    grad_scalar_i = self.projected_grads_list[i] 
                    lowdim_rge.add_(z_i * grad_scalar_i)
                
                lowdim_rge.div_(num_samples)

                # if name not in self.period_z:
                #     self.period_z[name] = lowdim_rge.clone()
                # else:
                #     self.period_z[name].add_(lowdim_rge)
                
                if name not in self.momentum:
                    self.momentum[name] = lowdim_rge.clone()
                else:
                    self.momentum[name].mul_(args.beta).add_(lowdim_rge)

                if args.momentum:
                    lowdim_rge = self.momentum[name]

                if opt_type == "muon":
                    M_sign = zeropower_via_newtonschulz5(lowdim_rge)
                    G = u @ M_sign @ v
                elif opt_type == "muon_svd":
                    M_sign = zeropower_via_svd(lowdim_rge)
                    G = u @ M_sign @ v
                else:
                    raise ValueError(f"Unsupported optimizer_type: {args.zo_optimizer}")
                
                if getattr(self, "subspace_update_mode", "gradient") == "gradient":
                    if name not in self.period_z:
                        self.period_z[name] = M_sign.clone()
                    else:
                        self.period_z[name].add_(M_sign)

                if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                    G.add_(param.data, alpha=args.weight_decay)

                param.data.add_(-lr * G)

            else:
                # --- 1D parameter update branch ---
                grad_est = torch.zeros_like(param.data)
                
                for i in range(num_samples):
                    z_i = self.z[name][i]
                    grad_scalar_i = self.projected_grads_list[i]
                    grad_est.add_(z_i * grad_scalar_i)
                
                grad_est.div_(num_samples)


                if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                    grad_est.add_(args.weight_decay, param.data)

                param.data.add_(-1e-7 * grad_est)

    def fulldim_zo_update_u(self, base_seed):
        args = self.args
        rank_r = args.rank_r
        energy_threshold = args.energy_threshold
        # p_keep = args.p_keep

        num_samples_MeZO = len(self.projected_grads_list_MeZO)

        for param_idx, (name, param) in enumerate(self.named_parameters_to_optim):
            if param.data.ndim >= 2:

                # exploit
                u_exploit, S_v, _ = torch.linalg.svd(self.momentum[name].to(device=param.data.device, dtype=torch.float32), full_matrices=False)
                p_keep_min = max(1, rank_r // 4)
                p_keep_max = max(1, int(0.75 * rank_r))
                energy = torch.cumsum(S_v ** 2, dim=0) / (torch.sum(S_v ** 2))
                threshold = torch.tensor(energy_threshold, device=energy.device, dtype=energy.dtype)
                p_keep_adapt = int(torch.searchsorted(energy, threshold).item() + 1)
                p_keep_adapt = min(max(p_keep_adapt, p_keep_min), p_keep_max)
                u_keep = self.u_matrices[name].float() @ u_exploit[:,:p_keep_adapt]
                u_keep, _ = torch.linalg.qr(u_keep, mode="reduced")

                # explore
                MeZO_grad = torch.zeros(param.data.shape[0], param.data.shape[1], device=param.data.device, dtype=torch.float32)
                for i in range(num_samples_MeZO):
                    current_seed = base_seed + i*1000000 + param_idx
                    torch.manual_seed(current_seed)
                    z_i = torch.randn_like(param.data) 
                    grad_scalar_i = self.projected_grads_list_MeZO[i] 
                    MeZO_grad.add_(z_i * grad_scalar_i)
                MeZO_grad.div_(num_samples_MeZO)
                explore_matrix = MeZO_grad - u_keep @ (u_keep.T @ MeZO_grad)
                u_explore, _, _ = torch.svd_lowrank(explore_matrix, q=rank_r-p_keep_adapt)
                u_total = torch.cat([u_keep, u_explore], dim=1)
                u_total, _ = torch.linalg.qr(u_total, mode="reduced")
                self.u_matrices[name] = u_total.to(param.device).to(param.dtype)

    ############## Misc overload functions ##############
    def _set_signature_columns_if_needed(self):
        """
        We overload this function for non-differentiable objective training to pass "gold" -- the gold text for the task
        """
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += list(set(["label", "label_ids"] + self.label_names))
            self._signature_columns += ["gold"]

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        if output_dir is None:
            output_dir = self.args.output_dir

        if is_torch_tpu_available():
            self._save_tpu(output_dir)
        elif self.args.should_save:
            self._save(output_dir)

        # Push to the Hub when `save_model` is called by the user.
        if self.args.push_to_hub and not _internal_call:
            self.push_to_hub(commit_message="Model save")

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        logging_steps = max(1, int(getattr(self.args, "logging_steps", 1)))
        if self.state.global_step % logging_steps == 0:
            logs: Dict[str, float] = {}
            gathered_tr_loss = self._nested_gather(tr_loss) if hasattr(self, "_nested_gather") else tr_loss
            tr_loss_scalar = gathered_tr_loss.mean().item()
            tr_loss -= tr_loss
            logs["train_loss"] = round(tr_loss_scalar / logging_steps, 4)
            logs["learning_rate"] = self._get_learning_rate()
            logs["eps"] = self.args.zo_eps
            logs["rank"] = self.args.rank_r
            if torch.cuda.is_available():
                # logs["gpu_memory_allocated_mb"] = round(torch.cuda.memory_allocated() / (1024 ** 2), 2)
                # logs["gpu_memory_reserved_mb"] = round(torch.cuda.memory_reserved() / (1024 ** 2), 2)
                logs["gpu_memory_max_allocated_mb"] = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)
            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()
            self.log(logs)
            maybe_log_to_wandb(logs, step=self.state.global_step)

        metrics = None
        should_evaluate = False
        eval_steps = getattr(self.args, "eval_steps", None)
        if self.args.evaluation_strategy == IntervalStrategy.STEPS and eval_steps:
            should_evaluate = self.state.global_step % eval_steps == 0

        if should_evaluate:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            merged_wandb_metrics = dict(metrics)
            # Validation-set task metrics drive best-checkpoint selection.
            val_samples = getattr(self, "raw_dev_samples", None)
            if val_samples is None:
                val_samples = getattr(self, "raw_eval_samples", None)
            if hasattr(self, "framework") and val_samples is not None:
                val_task_metrics = self.framework.evaluate([], val_samples)
                val_task_metrics = {
                    key: value.item() if isinstance(value, np.generic) else value
                    for key, value in val_task_metrics.items()
                }
                task_log = {}
                for key, value in val_task_metrics.items():
                    metric_key = key if key.startswith("eval_") else f"eval_{key}"
                    metrics[metric_key] = value
                    merged_wandb_metrics[key] = value
                    task_log[metric_key] = value
                if task_log:
                    self.log(task_log)
                checkpoint_improved = self._maybe_update_best_checkpoint(model, trial, metrics)
                self._maybe_stop_for_overfit(metrics, checkpoint_improved)
            maybe_log_to_wandb(prefix_metric_keys(merged_wandb_metrics, "eval/"), step=self.state.global_step)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

            # Run delayed LR scheduler now that metrics are populated
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                metric_to_check = self.args.metric_for_best_model
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                self.lr_scheduler.step(metrics[metric_to_check])

        should_save = False
        save_steps = getattr(self.args, "save_steps", None)
        if self.args.save_strategy == IntervalStrategy.STEPS and save_steps:
            should_save = self.state.global_step % save_steps == 0

        # Only keep/update checkpoint-best (via _maybe_update_best_checkpoint on eval).
        # Skip periodic checkpoint-{step} saves to avoid duplicating large model dumps.
        if should_save:
            logger.info(
                f"Skip periodic save at step {self.state.global_step}; "
                "only checkpoint-best is retained when validation improves."
            )

    def _maybe_update_best_checkpoint(self, model, trial, metrics):
        """Overwrite checkpoint-best whenever validation metric improves."""
        if metrics is None:
            return False

        metric_to_check = self.args.metric_for_best_model
        if metric_to_check is None:
            if "eval_accuracy" in metrics:
                metric_to_check = "accuracy"
            elif "eval_f1" in metrics:
                metric_to_check = "f1"
            elif "eval_loss" in metrics:
                metric_to_check = "loss"
            else:
                return False

        if not metric_to_check.startswith("eval_"):
            metric_key = f"eval_{metric_to_check}"
        else:
            metric_key = metric_to_check

        if metric_key not in metrics:
            for fallback_key in ("eval_accuracy", "eval_f1", "eval_loss"):
                if fallback_key in metrics:
                    logger.warning(
                        f"metric_for_best_model '{metric_key}' not found; falling back to '{fallback_key}'"
                    )
                    metric_key = fallback_key
                    metric_to_check = fallback_key
                    break
            else:
                logger.warning(
                    f"Cannot update best checkpoint: {metric_key} not in metrics {list(metrics.keys())}"
                )
                return False

        self._last_checkpoint_metric_key = metric_key
        metric_value = metrics[metric_key]
        greater_is_better = self.args.greater_is_better
        if greater_is_better is None:
            greater_is_better = metric_key not in ("loss", "eval_loss")
        operator = np.greater if greater_is_better else np.less

        if (
            self.state.best_metric is None
            or self.state.best_model_checkpoint is None
            or operator(metric_value, self.state.best_metric)
        ):
            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, "checkpoint-best")
            os.makedirs(output_dir, exist_ok=True)
            self.save_model(output_dir, _internal_call=True)
            self.state.best_metric = metric_value
            self.state.best_model_checkpoint = output_dir
            self.best_checkpoint_step = self.state.global_step
            if self.args.should_save:
                self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))
            logger.info(
                f"Updated best checkpoint at step {self.state.global_step}: "
                f"{metric_key}={metric_value} -> {output_dir}"
            )
            return True
        return False

    def _maybe_stop_for_overfit(self, metrics, checkpoint_improved):
        """Stop after sustained validation degradation.

        DROP's task metric can degrade substantially before language-model loss
        reaches the relatively high loss-ratio guard.  Treat either a clear
        metric drop or the configured loss-ratio increase as sufficient once
        the patience/min-step guards are met; both signals are still logged so
        callers can distinguish the reason for stopping.
        """
        patience = int(getattr(self.args, "early_stopping_patience", 0) or 0)
        if patience <= 0 or not metrics:
            return

        metric_key = getattr(self, "_last_checkpoint_metric_key", None)
        if metric_key is None or metric_key not in metrics or "eval_loss" not in metrics:
            return

        metric_value = float(metrics[metric_key])
        eval_loss = float(metrics["eval_loss"])
        if not math.isfinite(metric_value) or not math.isfinite(eval_loss):
            return

        min_eval_loss = getattr(self, "_early_stopping_min_eval_loss", None)
        if min_eval_loss is None or eval_loss < min_eval_loss:
            min_eval_loss = eval_loss
            self._early_stopping_min_eval_loss = eval_loss

        if checkpoint_improved:
            self._early_stopping_bad_evals = 0
        else:
            self._early_stopping_bad_evals = getattr(self, "_early_stopping_bad_evals", 0) + 1

        greater_is_better = self.args.greater_is_better
        if greater_is_better is None:
            greater_is_better = metric_key not in ("loss", "eval_loss")
        best_metric = float(self.state.best_metric)
        metric_drop = (
            best_metric - metric_value if greater_is_better else metric_value - best_metric
        )
        loss_ratio = eval_loss / max(min_eval_loss, 1e-12)
        min_steps = int(getattr(self.args, "early_stopping_min_steps", 0) or 0)
        required_metric_drop = float(
            getattr(self.args, "early_stopping_metric_drop", 0.02) or 0.0
        )
        required_loss_ratio = float(
            getattr(self.args, "early_stopping_loss_ratio", 1.15) or 1.0
        )

        logger.info(
            "Early-stop monitor at step %s: bad_evals=%s/%s, %s=%.6f, "
            "best=%.6f, metric_drop=%.6f, eval_loss=%.6f, min_eval_loss=%.6f, "
            "loss_ratio=%.4f",
            self.state.global_step,
            self._early_stopping_bad_evals,
            patience,
            metric_key,
            metric_value,
            best_metric,
            metric_drop,
            eval_loss,
            min_eval_loss,
            loss_ratio,
        )

        metric_overfit = metric_drop >= required_metric_drop
        loss_overfit = loss_ratio >= required_loss_ratio
        severe_overfit = (
            self.state.global_step >= min_steps
            and self._early_stopping_bad_evals >= patience
            and (metric_overfit or loss_overfit)
        )
        if severe_overfit:
            trigger = "metric_drop" if metric_overfit else "loss_ratio"
            logger.warning(
                "EARLY_STOP_OVERFIT step=%s best_step=%s bad_evals=%s "
                "metric_drop=%.6f loss_ratio=%.4f trigger=%s",
                self.state.global_step,
                getattr(self, "best_checkpoint_step", None),
                self._early_stopping_bad_evals,
                metric_drop,
                loss_ratio,
                trigger,
            )
            # Set both the public TrainerControl flag and a private persistent
            # flag.  Some evaluation/callback paths replace ``self.control``;
            # the private flag guarantees the custom loop exits immediately
            # after this evaluation and reaches final metric/JSON writing.
            self.control.should_training_stop = True
            self.control.should_epoch_stop = True
            self._stop_requested = True

    def _save_checkpoint(self, model, trial, metrics=None):
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # Save model checkpoint
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        if self.hp_search_backend is None and trial is None:
            self.store_flos()

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        self.save_model(output_dir, _internal_call=True)
        if self.deepspeed:
            # under zero3 model file itself doesn't get saved since it's bogus! Unless deepspeed
            # config `stage3_gather_16bit_weights_on_model_save` is True
            self.deepspeed.save_checkpoint(output_dir)

        if is_torch_tpu_available():
            xm.rendezvous("saving_optimizer_states")
            xm.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))
            with warnings.catch_warnings(record=True) as caught_warnings:
                xm.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
                reissue_pt_warnings(caught_warnings)
        elif is_sagemaker_mp_enabled():
            opt_state_dict = self.optimizer.local_state_dict(gather_if_shard=False)
            smp.barrier()
            if smp.rdp_rank() == 0 or smp.state.cfg.shard_optimizer_state:
                smp.save(
                    opt_state_dict,
                    os.path.join(output_dir, OPTIMIZER_NAME),
                    partial=True,
                    v3=smp.state.cfg.shard_optimizer_state,
                )
            if self.args.should_save:
                with warnings.catch_warnings(record=True) as caught_warnings:
                    torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
                reissue_pt_warnings(caught_warnings)
                if self.do_grad_scaling:
                    torch.save(self.scaler.state_dict(), os.path.join(output_dir, SCALER_NAME))
        elif self.args.should_save and not self.deepspeed:
            # deepspeed.save_checkpoint above saves model/optim/sched
            torch.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))
            with warnings.catch_warnings(record=True) as caught_warnings:
                torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
            reissue_pt_warnings(caught_warnings)
            if self.do_grad_scaling:
                torch.save(self.scaler.state_dict(), os.path.join(output_dir, SCALER_NAME))

        # Best-checkpoint tracking is handled in _maybe_update_best_checkpoint during evaluation.

        # Save the Trainer state
        if self.args.should_save:
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

        # Save RNG state in non-distributed training
        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            if self.args.local_rank == -1:
                # In non distributed, we save the global CUDA RNG state (will take care of DataParallel)
                rng_states["cuda"] = torch.cuda.random.get_rng_state_all()
            else:
                rng_states["cuda"] = torch.cuda.random.get_rng_state()

        if is_torch_tpu_available():
            rng_states["xla"] = xm.get_rng_state()

        # A process can arrive here before the process 0 has a chance to save the model, in which case output_dir may
        # not yet exist.
        os.makedirs(output_dir, exist_ok=True)

        if self.args.world_size <= 1:
            torch.save(rng_states, os.path.join(output_dir, "rng_state.pth"))
        else:
            torch.save(rng_states, os.path.join(output_dir, f"rng_state_{self.args.process_index}.pth"))

        if self.args.push_to_hub:
            self._push_from_checkpoint(output_dir)

        # Maybe delete some older checkpoints.
        # Do not rotate away checkpoint-best (non checkpoint-<step> dirs are ignored by HF rotation).
        if self.args.should_save and hasattr(self, "_rotate_checkpoints"):
            self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)
