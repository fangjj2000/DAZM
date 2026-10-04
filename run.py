import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import argparse
import os
import time
import tasks
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, Trainer, HfArgumentParser, Trainer, TrainingArguments, DataCollatorWithPadding, DataCollatorForTokenClassification
from typing import Union, Optional
import torch
from torch.nn.parameter import Parameter
import numpy as np
from dataclasses import dataclass, is_dataclass, asdict
from tqdm import tqdm
from tasks import get_task
import json
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from metrics import calculate_metric
from gradient_spectrum import GradientSpectrumRecorder
from utils import *
from utils_compat import maybe_log_to_wandb, prefix_metric_keys
from MeZOTrainer import MeZOTrainer
from LowDimTrainer import LowDimTrainer
from LOZOTrainer import LOZOTrainer
from SubZeroTrainer import SubZeroTrainer
from GreedyZounDimTrainer import GreedyZuonTrainer
from GELOZOtrainer import GELowRankTrainer
from transformers import (
    TrainerCallback, 
    TrainerControl, 
    TrainerState, 
)
import random
@dataclass
class OurArguments(TrainingArguments):
    # dataset and sampling strategy
    task_name: str = "SST2" # task name should match the string before Dataset in the Dataset class name. We support the following task_name: SST2, RTE, CB, BoolQ, WSC, WIC, MultiRC, Copa, ReCoRD, SQuAD, DROP
    trainer: str= "hessian"
    # Number of examples
    num_train: int = 0 # ICL mode: number of demonstrations; training mode: number of training samples
    num_dev: int = None # (only enabled with training) number of development samples
    num_eval: int = None # number of evaluation samples
    num_train_sets: int = None # how many sets of training samples/demos to sample; if None and train_set_seed is None, then we will sample one set for each evaluation sample
    train_set_seed: int = None # designated seed to sample training samples/demos
    result_file: str = None # file name for saving performance; if None, then use the task name, model name, and config
    do_grad_scaling: bool = False

    # Model loading
    model_name: str = "facebook/opt-125m" # HuggingFace model name
    model_path: str = None # local checkpoint dir (e.g. .../checkpoint-best); overrides model_name for loading weights/tokenizer
    load_float16: bool = False # load model parameters as float16
    load_bfloat16: bool = False # load model parameters as bfloat16
    load_int8: bool = False # load model parameters as int8
    max_length: int = 2048 # max length the model can take
    no_auto_device: bool = False # do not load model by auto device; should turn this on when using FSDP

    # Calibration
    sfc: bool = False # whether to use SFC calibration
    icl_sfc: bool = False # whether to use SFC calibration for ICL samples
    zo_optimizer: str = "sgd"  # "sgd" or "adam"
    multiple_sample: bool=False
    momentum: bool=False
    beta: float=0.9
    early_stopping_patience: int=0
    early_stopping_min_steps: int=0
    early_stopping_metric_drop: float=0.02
    early_stopping_loss_ratio: float=1.15
    max_time: int=0
    num_samples: int=4
    num_samples_MeZO: int=5
    step_interval: int = 50 
    step_u_interval: int = 50
    rank_r: int = 64 
    p: int = 32
    p_keep: int = 4
    energy_threshold: float = 0.8
    subspace_update_mode: str = "gradient" # gradient, frozen, or random
    capture_displacement: bool = False
    capture_layers: str = ""
    capture_refresh_stride: int = 5
    capture_output_file: str = None
    sparsity: float=1.0
    k_interval: int = 20000
    k_start: int = 4
    reset_v_on_p_refresh: bool = True
    phase2_steps: int = 2000
    zo_muon_beta: float = 0.9
    zo_perturbation_mode: str = "one_side"
    one_d_lr: float = 1e-7
    # Training
    trainer: str = "none" 
    ## options
    ## - none: no training -- for zero-shot or in-context learning (ICL)
    ## - regular: regular huggingface trainer -- for fine-tuning
    ## - zo: zeroth-order (MeZO) training
    only_train_option: bool = True # whether to only train the option part of the input
    train_as_classification: bool = False # take the log likelihood of all options and train as classification 

    # MeZO
    zo_eps: float = 1e-3 # eps in MeZO
    save_gradient_spectra: bool = False
    svd_save_interval: int = 100
    svd_max_layers: int = 4
    svd_max_values: int = 256
    svd_layers: str = ""
    svd_output_dir: str = None
    
    ###############diag
    warmup_step: int = 0
    decay_step: int = 0
    zo_lr_scheduler_type: str = 'constant'
    weight_decay: float = 0
    hessian_smooth_type: str = 'constant0'

    # Prefix tuning
    prefix_tuning: bool = False # whether to use prefix tuning
    num_prefix: int = 5 # number of prefixes to use
    no_reparam: bool = True # do not use reparameterization trick
    prefix_init_by_real_act: bool = True # initialize prefix by real activations of random words

    # LoRA
    lora: bool = False # whether to use LoRA
    lora_alpha: int = 16 # alpha in LoRA
    lora_r: int = 8 # r in LoRA

    # Generation
    sampling: bool = False # whether to use sampling
    temperature: float = 1.0 # temperature for generation
    num_beams: int = 1 # number of beams for generation
    top_k: int = None # top-k for generation
    top_p: float = 0.95 # top-p for generation
    max_new_tokens: int = 50 # max number of new tokens to generate
    eos_token: str = "\n" # end of sentence token

    # Saving
    save_model: bool = False # whether to save the model
    no_eval: bool = False # whether to skip evaluation
    tag: str = "" # saving tag

    # Linear probing
    linear_probing: bool = False # whether to do linear probing
    lp_early_stopping: bool = False # whether to do early stopping in linear probing
    head_tuning: bool = False # head tuning: only tune the LM head

    # Untie emb/lm_head weights
    untie_emb: bool = False # untie the embeddings and LM head

    # Display
    verbose: bool = False # verbose output

    # Non-diff objective
    non_diff: bool = False # use non-differentiable objective (only support F1 for SQuAD for now)

    # Auto saving when interrupted
    save_on_interrupt: bool = False # save model when interrupted (useful for long training)

    # Trainer compatibility
    past_index: int = -1 # keep HF Trainer compatibility for models that may expose cached past states
    overwrite_output_dir: bool = False # HF 5.x removed this field; keep CLI compatibility with existing scripts


def parse_args():
    parser = argparse.ArgumentParser()
    parser = HfArgumentParser(OurArguments)
    args = parser.parse_args_into_dataclasses()[0]
    print(args)
    return args


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _to_wandb_safe_metrics(metrics):
    safe_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, np.generic):
            safe_metrics[key] = value.item()
        else:
            safe_metrics[key] = value
    return safe_metrics


def _format_duration(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours}h {minutes}m {secs:.2f}s"


def _sanitize_metric_value(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def _collect_split_metrics(framework, samples, prefix):
    split_metrics = framework.evaluate([], samples)
    return {f"{prefix}{key}": _sanitize_metric_value(value) for key, value in split_metrics.items()}


def _snapshot_param(model, preferred_keys=None):
    """Capture a small CPU snapshot for load verification (prefer trained layer weights)."""
    named = dict(model.named_parameters())
    if preferred_keys:
        for key in preferred_keys:
            if not key:
                continue
            param = named.get(key)
            if param is not None and param.numel() > 0:
                return key, param.detach().float().reshape(-1)[:64].cpu().clone()

    for name, param in model.named_parameters():
        if (
            param is not None
            and param.numel() > 0
            and param.ndim >= 2
            and "layers." in name
            and param.requires_grad
        ):
            return name, param.detach().float().reshape(-1)[:64].cpu().clone()

    for key in (
        "model.decoder.embed_tokens.weight",
        "model.embed_tokens.weight",
        "model.decoder.embed_positions.weight",
    ):
        param = named.get(key)
        if param is not None and param.numel() > 0:
            return key, param.detach().float().reshape(-1)[:64].cpu().clone()

    for name, param in model.named_parameters():
        if param is not None and param.numel() > 0:
            return name, param.detach().float().reshape(-1)[:64].cpu().clone()
    return None, None


def _snapshot_from_checkpoint(checkpoint_dir, preferred_key=None):
    """Read the same leading slice from a checkpoint tensor (match live-model key when possible)."""
    import os
    from safetensors import safe_open

    index_path = os.path.join(checkpoint_dir, "model.safetensors.index.json")
    single_path = os.path.join(checkpoint_dir, "model.safetensors")

    def _pick_key(available):
        if preferred_key is not None and preferred_key in available:
            return preferred_key
        for key in available:
            if "layers." in key and key.endswith(".weight"):
                return key
        for key in (
            "model.decoder.embed_tokens.weight",
            "model.embed_tokens.weight",
            "model.decoder.embed_positions.weight",
        ):
            if key in available:
                return key
        return next(iter(available))

    if os.path.isfile(index_path):
        weight_map = json.load(open(index_path))["weight_map"]
        key = _pick_key(weight_map)
        shard = weight_map[key]
        with safe_open(os.path.join(checkpoint_dir, shard), framework="pt", device="cpu") as f:
            return key, f.get_tensor(key).float().reshape(-1)[:64].clone()
    if os.path.isfile(single_path):
        with safe_open(single_path, framework="pt", device="cpu") as f:
            key = _pick_key(list(f.keys()))
            return key, f.get_tensor(key).float().reshape(-1)[:64].clone()
    return None, None


def _manual_load_checkpoint_weights(model, checkpoint_dir):
    """Copy safetensors/bin weights onto the live model parameters (device_map-safe).

    Returns the number of tensors successfully copied.
    """
    import gc
    import os

    index_path = os.path.join(checkpoint_dir, "model.safetensors.index.json")
    single_path = os.path.join(checkpoint_dir, "model.safetensors")
    bin_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
    named_params = dict(model.named_parameters())
    named_buffers = dict(model.named_buffers())

    def _copy_tensor(key, tensor):
        target = named_params.get(key, named_buffers.get(key))
        if target is None:
            return False
        with torch.no_grad():
            target.copy_(tensor.to(device=target.device, dtype=target.dtype))
        return True

    if os.path.isfile(index_path):
        from safetensors.torch import load_file
        weight_map = json.load(open(index_path))["weight_map"]
        shard_to_keys = {}
        for key, shard in weight_map.items():
            shard_to_keys.setdefault(shard, []).append(key)
        copied = 0
        for shard, keys in shard_to_keys.items():
            state = load_file(os.path.join(checkpoint_dir, shard), device="cpu")
            for key in keys:
                if key in state and _copy_tensor(key, state[key]):
                    copied += 1
            del state
            gc.collect()
        if copied == 0:
            raise RuntimeError(f"No overlapping tensors found when loading {checkpoint_dir}")
        logger.info(f"Loaded sharded safetensors checkpoint ({copied} tensors): {checkpoint_dir}")
        return copied

    if os.path.isfile(single_path):
        from safetensors.torch import load_file
        state = load_file(single_path, device="cpu")
        copied = sum(1 for key, tensor in state.items() if _copy_tensor(key, tensor))
        del state
        gc.collect()
        if copied == 0:
            raise RuntimeError(f"No overlapping tensors found when loading {checkpoint_dir}")
        logger.info(f"Loaded safetensors checkpoint ({copied} tensors): {checkpoint_dir}")
        return copied

    if os.path.isfile(bin_path):
        state = torch.load(bin_path, map_location="cpu")
        copied = sum(1 for key, tensor in state.items() if _copy_tensor(key, tensor))
        del state
        gc.collect()
        if copied == 0:
            raise RuntimeError(f"No overlapping tensors found when loading {checkpoint_dir}")
        logger.info(f"Loaded pytorch_model.bin checkpoint ({copied} tensors): {checkpoint_dir}")
        return copied

    raise FileNotFoundError(
        f"No loadable weights found in {checkpoint_dir} "
        "(expected model.safetensors.index.json / model.safetensors / pytorch_model.bin)"
    )


def _tensors_close(a, b):
    if a is None or b is None:
        return False
    return torch.allclose(a, b.to(dtype=a.dtype), rtol=1e-3, atol=1e-3)


def _load_checkpoint_weights(model, checkpoint_dir, expect_change=True):
    """
    Load weights from a (possibly sharded) checkpoint into an existing model.

    HF Trainer._load_best_model() / accelerate.load_checkpoint_in_model can silently
    no-op on device_map='auto' models. We verify a parameter snapshot and fall back to
    an explicit tensor copy when needed.
    """
    import os

    if checkpoint_dir is None or not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    snap_key, before = _snapshot_param(model)
    used_manual = False
    copied = 0

    # Try accelerate first for convenience, then verify.
    try:
        from accelerate.utils import load_checkpoint_in_model
        device_map = getattr(model, "hf_device_map", None)
        load_checkpoint_in_model(model, checkpoint_dir, device_map=device_map)
        logger.info(f"Tried accelerate.load_checkpoint_in_model: {checkpoint_dir}")
    except Exception as exc:
        logger.warning(
            f"accelerate.load_checkpoint_in_model failed ({type(exc).__name__}: {exc}); "
            "falling back to manual safetensors load."
        )
        copied = _manual_load_checkpoint_weights(model, checkpoint_dir)
        used_manual = True

    _, after = _snapshot_param(model, preferred_keys=(snap_key,) if snap_key else None)
    changed = not _tensors_close(before, after)

    if not changed and not used_manual:
        logger.warning(
            "accelerate load left model weights unchanged; falling back to manual safetensors copy."
        )
        copied = _manual_load_checkpoint_weights(model, checkpoint_dir)
        used_manual = True
        _, after = _snapshot_param(model, preferred_keys=(snap_key,) if snap_key else None)
        changed = not _tensors_close(before, after)

    if changed or (used_manual and copied > 0):
        return True

    disk_key, disk = _snapshot_from_checkpoint(checkpoint_dir, preferred_key=snap_key)
    if _tensors_close(after, disk):
        logger.info(
            f"Model already matched checkpoint weights at {checkpoint_dir} "
            f"(key={disk_key or snap_key}); nothing to update."
        )
        return True

    msg = (
        f"Checkpoint load from {checkpoint_dir} appears to have left model weights unchanged. "
        "Best evaluation would be invalid."
    )
    if expect_change:
        raise RuntimeError(msg)
    logger.warning(msg)
    return False


def _checkpoint_step_from_path(checkpoint_dir):
    """Parse trailing step from .../checkpoint-1234 or .../checkpoint-best."""
    import os
    import re

    if not checkpoint_dir:
        return None
    name = os.path.basename(os.path.normpath(checkpoint_dir))
    match = re.fullmatch(r"checkpoint-(\d+)", name)
    if match:
        return int(match.group(1))
    return None


def _resolve_best_checkpoint_step(trainer, best_ckpt):
    if trainer is not None:
        step = getattr(trainer, "best_checkpoint_step", None)
        if step is not None:
            return int(step)
        # HF trainer_state may record best_metric step indirectly via checkpoint path.
        state_step = getattr(getattr(trainer, "state", None), "best_global_step", None)
        if state_step is not None:
            return int(state_step)
    parsed = _checkpoint_step_from_path(best_ckpt)
    if parsed is not None:
        return parsed
    return None


class MaxTimeCallback(TrainerCallback):
    """
    A callback that stops training after a specified number of seconds.
    """
    def __init__(self, max_seconds):
        self.start_time = time.time()
        self.max_seconds = max_seconds

    def on_step_end(self, args, state, control, **kwargs):
        elapsed = time.time() - self.start_time
        if elapsed > self.max_seconds:
            print(f"\n[MaxTimeCallback] Time limit of {self.max_seconds}s reached. Stopping training.")
            control.should_training_stop = True # <--- THIS WAS MISSING OR IMPLIED
            return control



from transformers import Trainer

class RegularTrainer(Trainer):
    """
    A custom Trainer that fixes the compatibility issue between standard AdamW 
    and the modified transformers library used in ZO/MeZO environments.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gradient_spectrum_recorder = GradientSpectrumRecorder(self.args, self.model, method_name="fo")
        self._spectrum_micro_step = 0

    def create_optimizer(self):
        # Create the standard optimizer (AdamW) using the parent class logic
        optimizer = super().create_optimizer()
        
        # Monkey-patch: Add a dummy 'train' method if it doesn't exist.
        if not hasattr(optimizer, "train"):
            optimizer.train = lambda: None

        # Monkey-patch: Add a dummy 'eval' method if it doesn't exist.
        # This prevents the AttributeError: 'AdamW' object has no attribute 'eval'
        if not hasattr(optimizer, "eval"):
            optimizer.eval = lambda: None
            
        return optimizer

    def training_step(self, model, inputs, *inner_args, **inner_kwargs):
        loss = super().training_step(model, inputs, *inner_args, **inner_kwargs)
        self._spectrum_micro_step += 1
        grad_accum_steps = max(1, int(getattr(self.args, "gradient_accumulation_steps", 1)))
        logical_step = self.state.global_step + 1
        if (
            self.gradient_spectrum_recorder.enabled
            and self._spectrum_micro_step % grad_accum_steps == 0
            and self.gradient_spectrum_recorder.should_capture(logical_step)
        ):
            named_tensors = {}
            for name, param in model.named_parameters():
                if param.grad is None or not self.gradient_spectrum_recorder.is_selected(name):
                    continue
                named_tensors[name] = param.grad.detach()
            self.gradient_spectrum_recorder.capture_from_named_tensors(
                step=logical_step,
                source="first_order_gradient",
                named_tensors=named_tensors,
            )
        return loss

class Framework:

    def __init__(self, args, task):
        self.args = args
        self.task = task
        self.model, self.tokenizer = self.load_model()


    def load_model(self):
        """
        Load HuggingFace models.

        If --model_path is set, load config/weights/tokenizer from that local checkpoint
        (e.g. result_uv/.../checkpoint-best). Otherwise load from --model_name.
        """
        import os

        pretrained_path = self.args.model_path or self.args.model_name
        if self.args.model_path:
            if not os.path.isdir(self.args.model_path):
                raise FileNotFoundError(f"--model_path not found: {self.args.model_path}")
            logger.info(f"Loading model from local checkpoint: {self.args.model_path}")

        with count_time("Loading model with FP%d" % (16 if self.args.load_float16 else 32)):
            free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
            config = AutoConfig.from_pretrained(pretrained_path, trust_remote_code=True)
            # Keep tagging / name checks useful when only --model_path is provided.
            if self.args.model_path and (
                not self.args.model_name or self.args.model_name == "facebook/opt-125m"
            ):
                inferred = getattr(config, "_name_or_path", None) or pretrained_path
                self.args.model_name = inferred
            if self.args.untie_emb:
                # Untie embeddings/LM head
                logger.warn("Untie embeddings and LM head")
                config.tie_word_embeddings = False
            if self.args.head_tuning:
                # Head tuning
                from ht_opt import OPTForCausalLM
                model = OPTForCausalLM.from_pretrained(
                    pretrained_path,
                    config=config,
                )
            elif self.args.no_auto_device:
                # No auto device (use for FSDP)
                model = AutoModelForCausalLM.from_pretrained(
                    pretrained_path,
                    config=config,
                    trust_remote_code=True,
                )
            else:
                # Auto device loading
                torch_dtype = torch.float32
                if self.args.load_float16:
                    torch_dtype = torch.float16
                elif self.args.load_bfloat16:
                    torch_dtype = torch.bfloat16
                model_max_memory_gb = os.environ.get("MODEL_MAX_MEMORY_GB")
                if model_max_memory_gb:
                    try:
                        memory_limit = float(model_max_memory_gb)
                    except ValueError as exc:
                        raise ValueError(
                            f"MODEL_MAX_MEMORY_GB must be a positive number, got {model_max_memory_gb!r}"
                        ) from exc
                    if memory_limit <= 0:
                        raise ValueError(
                            f"MODEL_MAX_MEMORY_GB must be positive, got {model_max_memory_gb!r}"
                        )
                    max_memory = {
                        i: f"{memory_limit:g}GB" for i in range(torch.cuda.device_count())
                    }
                    logger.info("Using explicit per-GPU model memory limits: %s", max_memory)
                else:
                    max_memory = {
                        i: f'{free_in_GB-5}GB' for i in range(torch.cuda.device_count())
                    }
                model_kwargs = dict(
                    config=config,
                    device_map='auto',
                    torch_dtype=torch_dtype,
                    max_memory=max_memory,
                    trust_remote_code=True,
                )
                if self.args.load_int8:
                    model_kwargs["load_in_8bit"] = True
                model = AutoModelForCausalLM.from_pretrained(
                    pretrained_path,
                    **model_kwargs,
                )
                
                
            model.eval()

        # Load tokenizer (prefer checkpoint files when --model_path is set)
        tokenizer_path = pretrained_path
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False, trust_remote_code=True)

        tokenizer.pad_token_id = 0
        
        # HF tokenizer bug fix
        model_type = getattr(config, "model_type", "") or ""
        if "opt" in str(self.args.model_name).lower() or model_type == "opt":
            tokenizer.bos_token_id = 0

        # Prefix tuning/LoRA
        if self.args.prefix_tuning:
            from prefix import PrefixTuning
            PrefixTuning(model, num_prefix=self.args.num_prefix, reparam=not self.args.no_reparam, float16=self.args.load_float16, init_by_real_act=self.args.prefix_init_by_real_act)
        if self.args.lora:
            from lora import LoRA
            LoRA(model, r=self.args.lora_r, alpha=self.args.lora_alpha, float16=self.args.load_float16)

        if self.args.head_tuning:
            if model.config.model_type == "opt":
                head_name = "lm_head" if self.args.untie_emb else "embed_tokens"
            else:
                raise NotImplementedError
            for n, p in model.named_parameters():
                if head_name not in n:
                    p.requires_grad = False
                else:
                    logger.info(f"Only tuning {n}")

        return model, tokenizer

    

    def forward(self, input_ids, option_len=None, generation=False):
        """
        Given input_ids and the length of the option, return the log-likelihood of each token in the option.
        For generation tasks, return the generated text.
        This function is only for inference
        """
        input_ids = torch.tensor([input_ids]).to(self.model.device)

        if generation:
            args = self.args
            # Autoregressive generation
            outputs = self.model.generate(
                input_ids, do_sample=args.sampling, temperature=args.temperature, 
                num_beams=args.num_beams, top_p=args.top_p, top_k=args.top_k, max_new_tokens=min(args.max_new_tokens, args.max_length - input_ids.size(1)), 
                num_return_sequences=1, eos_token_id=[self.tokenizer.encode(args.eos_token, add_special_tokens=False)[0], self.tokenizer.eos_token_id],
            )
            # For generation, directly return the text output
            output_text = self.tokenizer.decode(outputs[0][input_ids.size(1):], skip_special_tokens=True).strip()
            return output_text
        else:
            with torch.inference_mode():
                self.model.eval()
                logits = self.model(input_ids=input_ids).logits
            labels = input_ids[0, 1:]
            logits = logits[0, :-1] 
            log_probs = F.log_softmax(logits, dim=-1)

            selected_log_probs = log_probs[torch.arange(len(labels)).to(labels.device), labels]
            selected_log_probs = selected_log_probs.cpu().detach()
            # Only return the option (candidate) part
            return selected_log_probs[-option_len:]


    def one_step_pred(self, train_samples, eval_sample, verbose=False):
        """
        Return the prediction on the eval sample. In ICL, use train_samples as demonstrations
        """
        verbose = verbose or self.args.verbose
        if verbose:
            logger.info("========= Example =========")
            logger.info(f"Candidate: {eval_sample.candidates}")
            logger.info(f"Correct candidate: {eval_sample.correct_candidate}")


        # Encode (add prompt and tokenize) the sample; if multiple-choice/classification, encode all candidates (options)
        encoded_candidates, option_lens = encode_prompt(
            self.task, self.task.get_template(), train_samples, eval_sample, self.tokenizer, max_length=self.args.max_length, 
            generation=self.task.generation, max_new_tokens=self.args.max_new_tokens
        )

        # Calibration
        if self.args.sfc or self.args.icl_sfc:
            sfc_encoded_candidates, sfc_option_lens = encode_prompt(self.task, self.task.get_template(), 
                train_samples, eval_sample, self.tokenizer, max_length=self.args.max_length,
                sfc=self.args.sfc, icl_sfc=self.args.icl_sfc, generation=self.task.generation, 
                max_new_tokens=self.args.max_new_tokens
            )

        outputs = []
        if self.task.generation:
            # For generation tasks, return the autoregressively-generated text
            output_text = self.forward(encoded_candidates[0], generation=True)
            if verbose:
                logger.info("=== Prompt ===")
                logger.info(self.tokenizer.decode(encoded_candidates[0]))
                logger.info(f"Output: {output_text}") 
            return Prediction(correct_candidate=eval_sample.correct_candidate, predicted_candidate=output_text)
        else:
            # For classification/multiple-choice, calculate the probabilities of all candidates
            for candidate_id, encoded_candidate in enumerate(encoded_candidates):
                selected_log_probs = self.forward(encoded_candidate, option_len=option_lens[candidate_id])
                if verbose:
                    if candidate_id == 0:
                        logger.info("=== Candidate %d ===" % candidate_id)
                        logger.info(self.tokenizer.decode(encoded_candidate))
                    else:
                        logger.info("=== Candidate %d (without context)===" % candidate_id)
                        logger.info(self.tokenizer.decode(encoded_candidate).split(self.task.train_sep)[-1])
                    logger.info(f"Log probabilities of the option tokens: {selected_log_probs}")

                if self.args.sfc or self.args.icl_sfc:
                    sfc_selected_log_probs = self.forward(sfc_encoded_candidates[candidate_id], option_len=sfc_option_lens[candidate_id])
                    if verbose:
                        logger.info("=== Candidate %d (without context) SFC ===" % candidate_id)
                        logger.info(self.tokenizer.decode(sfc_encoded_candidates[candidate_id]).split(self.task.train_sep)[-1])
                        logger.info(f"Log probabilities of the option tokens: {sfc_selected_log_probs}")

                outputs.append({"log_probs": selected_log_probs, "sfc_log_probs": sfc_selected_log_probs if self.args.sfc or self.args.icl_sfc else None})

            if self.args.sfc or self.args.icl_sfc:
                # Calibrated probabilities (surface form competition; https://arxiv.org/pdf/2104.08315.pdf)
                # log p(candidate | input) = log p_lm(candidate | input) - log p_lm(candidate | sfc prompt)
                scores = [x['log_probs'].sum().item() - x['sfc_log_probs'].sum().item() for x in outputs]
            else:
                # (Default) length-normalized log probabilities
                # log p(candidate | input) = log p_lm(candidate | input) / |candidate #tokens|
                scores = [x['log_probs'].mean().item() for x in outputs]

            if verbose:
                logger.info(f"Prediction scores: {scores}")

            if isinstance(eval_sample.correct_candidate, list):
                # For some datasets there are multiple correct answers
                correct_candidate_id = [eval_sample.candidates.index(c) for c in eval_sample.correct_candidate]
            else:
                correct_candidate_id = eval_sample.candidates.index(eval_sample.correct_candidate)

            return Prediction(correct_candidate=correct_candidate_id, predicted_candidate=int(np.argmax(scores)))


    def evaluate(self, train_samples, eval_samples, one_train_set_per_eval_sample=False):
        """
        Evaluate function. If one_train_set_per_eval_sample is True, then each eval sample has its own training (demonstration) set.
        """
        # if one_train_set_per_eval_sample:
        #     logger.info(f"There are {len(eval_samples)} validation samples and one train set per eval sample")
        # else:
        #     logger.info(f"There are {len(train_samples)} training samples and {len(eval_samples)} validation samples")

        # Prediction loop
        predictions = []  
        for eval_id, eval_sample in enumerate(tqdm(eval_samples)):
            predictions.append(
                self.one_step_pred(train_samples[eval_id] if one_train_set_per_eval_sample else train_samples, eval_sample)
            )

        # Calculate metrics 
        metric_name = getattr(self.task, "metric_name", "accuracy")
        metrics = {metric_name: calculate_metric(predictions, metric_name)}
        return metrics


    def train(self, train_samples, eval_samples):
        """
        Training function
        """
        # Set tokenizer to left padding (so that all the options are right aligned)
        self.tokenizer.padding_side = "left"

        class HFDataset(Dataset):

            def __init__(self, data):
                self.data = data

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]


        def _convert(samples):
            """
            Convert samples to HF-compatible dataset
            """
            data = []
            for sample in samples:
                encoded_candidates, option_lens = encode_prompt(
                    self.task, self.task.get_template(), [], sample, self.tokenizer, 
                    max_length=self.args.max_length, generation=self.task.generation, generation_with_gold=True, 
                    max_new_tokens=self.args.max_new_tokens
                )
                if self.task.generation:
                    correct_candidate_id = 0
                elif isinstance(sample.correct_candidate, list):
                    correct_candidate_id = sample.candidates.index(sample.correct_candidate[0])
                else:
                    correct_candidate_id = sample.candidates.index(sample.correct_candidate)
                
                if self.args.non_diff:
                    # For non-differentiable objective, there is no teacher forcing thus the 
                    # current answer part is removed
                    encoded_candidates[correct_candidate_id] = encoded_candidates[correct_candidate_id][:-option_lens[correct_candidate_id]]

                if self.args.train_as_classification:
                    # For classification, we provide the label as the correct candidate id
                    data.append([{"input_ids": encoded_candidates[_i], "labels": correct_candidate_id, "option_len": option_lens[_i], "num_options": len(sample.candidates)} for _i in range(len(encoded_candidates))])
                elif self.args.only_train_option:
                    # Otherwise, it is just LM-style teacher forcing
                    if self.args.non_diff:
                        # For non-differentiable objective, we need to provide the gold answer to calculate F1/acc
                        data.append({"input_ids": encoded_candidates[correct_candidate_id], "labels": encoded_candidates[correct_candidate_id], "option_len": option_lens[correct_candidate_id], "gold": sample.correct_candidate})
                    else:
                        data.append({"input_ids": encoded_candidates[correct_candidate_id], "labels": encoded_candidates[correct_candidate_id], "option_len": option_lens[correct_candidate_id]})
                else:
                    data.append({"input_ids": encoded_candidates[correct_candidate_id], "labels": encoded_candidates[correct_candidate_id]})
            return data

        with count_time("Tokenizing training samples"):
            train_dataset = HFDataset(_convert(train_samples))
            eval_dataset = HFDataset(_convert(eval_samples))

        if self.args.only_train_option and not self.args.non_diff:
            # If --only_train_option and not with a non-differentiable objective, we wrap the forward function
            self.model.original_forward = self.model.forward
            self.model.forward = forward_wrap_with_option_len.__get__(self.model, type(self.model))
        #########################################

        if self.args.non_diff:
            collator = NondiffCollator
        else:
            collator = DataCollatorForTokenClassification
        data_collator = DataCollatorWithPaddingAndNesting(self.tokenizer, pad_to_multiple_of=8) if self.args.train_as_classification else collator(self.tokenizer, pad_to_multiple_of=8)
        if self.args.trainer=="mezo":
            trainer = MeZOTrainer(
                model=self.model, 
                args=self.args,
                train_dataset=train_dataset, 
                eval_dataset=eval_dataset,
                data_collator=data_collator,
            )
            trainer.tokenizer = self.tokenizer
        elif self.args.trainer=="lowdim":
            trainer = LowDimTrainer(
                model=self.model, 
                args=self.args,
                train_dataset=train_dataset, 
                eval_dataset=eval_dataset,
                data_collator=data_collator,
            )
            trainer.tokenizer = self.tokenizer
        elif self.args.trainer=="lozo":
            trainer = LOZOTrainer(
                model=self.model, 
                args=self.args,
                train_dataset=train_dataset, 
                eval_dataset=eval_dataset,
                data_collator=data_collator,
            )
            trainer.tokenizer = self.tokenizer
        elif self.args.trainer=="subzero":
            trainer = SubZeroTrainer(
                model=self.model, 
                args=self.args,
                train_dataset=train_dataset, 
                eval_dataset=eval_dataset,
                data_collator=data_collator,
            )
            trainer.tokenizer = self.tokenizer
        elif self.args.trainer=="greedyzomuon":
            trainer = GreedyZuonTrainer(
                model=self.model, 
                args=self.args,
                train_dataset=train_dataset, 
                eval_dataset=eval_dataset,
                data_collator=data_collator,
            )
            trainer.tokenizer = self.tokenizer
        elif self.args.trainer in ("gelozo", "GELOZO"):
            trainer = GELowRankTrainer(
                model=self.model,
                args=self.args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
            )
            trainer.tokenizer = self.tokenizer
        else:
            # Use the custom RegularTrainer here
            trainer = RegularTrainer(
                model=self.model, 
                args=self.args,
                train_dataset=train_dataset, 
                eval_dataset=eval_dataset,
                data_collator=data_collator,
            )
            trainer.tokenizer = self.tokenizer
        trainer.framework = self
        # eval_samples here is the validation set used by HF evaluate / best-ckpt selection.
        trainer.raw_dev_samples = eval_samples
        # Optional held-out test set for final reporting (set in main as framework.test_samples).
        trainer.raw_eval_samples = getattr(self, "test_samples", eval_samples)
        self.trainer = trainer
        if self.args.save_on_interrupt:
            trainer.add_callback(SIGUSR1Callback())
        
        # ==========================================
        # NEW: Inject MaxTimeCallback if requested
        # ==========================================
        if hasattr(self.args, "max_time") and self.args.max_time is not None and self.args.max_time > 0:
            logger.info(f"Enabling MaxTimeCallback: Training will stop after {self.args.max_time} seconds.")
            trainer.add_callback(MaxTimeCallback(max_seconds=self.args.max_time))
        
        # Resume training from a last checkpoint
        last_checkpoint = None
        from transformers.trainer_utils import get_last_checkpoint
        overwrite_output_dir = getattr(self.args, "overwrite_output_dir", False)
        if os.path.isdir(self.args.output_dir) and not overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(self.args.output_dir)
        if last_checkpoint is not None and self.args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
        if self.args.resume_from_checkpoint is not None:
            last_checkpoint = self.args.resume_from_checkpoint

        # Train loop
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint) 
        
        # Log completion
        logger.info(f"Training finished. Global step: {train_result.global_step}")

        # Explicitly save the model
        if self.args.save_model:
            logger.warn("Save model..")
            trainer.save_model()
        
        # FSDP compatibility
        self.model = trainer.model 
        
        # Reset the forward function for evaluation
        if self.args.only_train_option and not self.args.non_diff:
            if type(self.model) == FSDP:
                logger.info("This is an FSDP model now. Be careful when assigning back the original forward function")
                self.model._fsdp_wrapped_module.forward = self.model._fsdp_wrapped_module.original_forward
            else:
                self.model.forward = self.model.original_forward

def result_file_tag(args):
    """
    Get the result file tag
    """
    save_model_name = args.model_name.split("/")[-1]
    sfc_tag = "-sfc" if args.sfc else ""
    icl_sfc_tag = "-icl_sfc" if args.icl_sfc else ""
    sample_eval_tag = "-sampleeval%d" % args.num_eval if args.num_eval is not None else ""
    sample_train_tag = "-ntrain%d" % args.num_train if args.num_train > 0 else ""
    sample_dev_tag = "-ndev%d" % args.num_dev if args.num_dev is not None else ""
    customized_tag = f"-{args.tag}" if len(args.tag) > 0 else ""
    return f"{args.task_name}-{save_model_name}" + sfc_tag + icl_sfc_tag + sample_eval_tag + sample_train_tag + sample_dev_tag + customized_tag


# def main():
#     args = parse_args()

#     set_seed(args.seed)
#     task = get_task(args.task_name)
#     train_sets = task.sample_train_sets(num_train=args.num_train, num_dev=args.num_dev, num_eval=args.num_eval, num_train_sets=args.num_train_sets, seed=args.train_set_seed)

#     # Initialize trainer and load model
#     framework = Framework(args, task)
#     if args.train_set_seed is not None or args.num_train_sets is not None:
#         # Eval samples share one (or multiple) training set(s)
#         for train_set_id, train_samples in enumerate(train_sets):
#             train_set_seed = train_set_id if args.train_set_seed is None else args.train_set_seed

#             # Sample eval samples
#             if args.num_eval is not None:
#                 eval_samples = task.sample_subset(data_split="valid", seed=train_set_seed, num=args.num_eval)
#             else:
#                 eval_samples = task.valid_samples

#             if args.trainer != "none":
#                 if args.num_dev is not None:
#                     # Dev samples
#                     dev_samples = train_samples[-args.num_dev:] 
#                     train_samples = train_samples[:-args.num_dev]
#                 else:
#                     dev_samples = None

#                 # Training
#                 framework.train(train_samples, dev_samples if dev_samples is not None else eval_samples)

#                 if not args.no_eval:
#                     metrics = framework.evaluate([], eval_samples) # No in-context learning if there is training
#                     if dev_samples is not None:
#                         dev_metrics = framework.evaluate([], dev_samples) 
#                         for m in dev_metrics:
#                             metrics["dev_" + m] = dev_metrics[m]
#             else:
#                 assert args.num_dev is None
#                 # Zero-shot / in-context learning
#                 metrics = framework.evaluate(train_samples, eval_samples)

#             if not args.no_eval:
#                 logger.info("===== Train set %d =====" % train_set_seed)
#                 logger.info(metrics)
#                 if args.local_rank <= 0:
#                     write_metrics_to_file(metrics, "result/" +  result_file_tag(args) + f"-trainset{train_set_id}.json" if args.result_file is None else args.result_file)

#     else:
#         # For each eval sample, there is a training set. no training is allowed
#         # This is for in-context learning (ICL)
#         assert args.trainer == "none"
#         if args.num_eval is not None:
#             eval_samples = task.sample_subset(data_split="valid", seed=0, num=args.num_eval)
#         else:
#             eval_samples = task.valid_samples

#         metrics = framework.evaluate(train_sets, eval_samples, one_train_set_per_eval_sample=True)
#         logger.info(metrics)

# if __name__ == "__main__": 
#     main()

import time
import logging

# Assuming other necessary imports (like parse_args, set_seed, etc.) are already present in your file.
# If logger is not defined globally, you might need to initialize it. 
# For this snippet, I assume 'logger' is available as per your original code.

def main():
    args = parse_args()

    set_seed(args.seed)
    task = get_task(args.task_name)
    train_sets = task.sample_train_sets(num_train=args.num_train, num_dev=args.num_dev, num_eval=args.num_eval, num_train_sets=args.num_train_sets, seed=args.train_set_seed)

    # Initialize trainer and load model
    framework = Framework(args, task)

    # --- Start Timer ---
    total_start_time = time.time()
    train_duration = None
    result_path = None
    metrics = {}
    
    if args.train_set_seed is not None or args.num_train_sets is not None:
        # Eval samples share one (or multiple) training set(s)
        for train_set_id, train_samples in enumerate(train_sets):
            train_set_seed = train_set_id if args.train_set_seed is None else args.train_set_seed

            # Sample eval samples (held-out test split for this codebase)
            if args.num_eval is not None:
                eval_samples = task.sample_subset(data_split="valid", seed=train_set_seed, num=args.num_eval)
            else:
                eval_samples = task.valid_samples

            if args.trainer != "none":
                if args.num_dev is not None:
                    if len(train_samples) <= args.num_dev:
                        raise ValueError(
                            f"num_dev={args.num_dev} leaves no training samples for task "
                            f"{args.task_name} (sampled {len(train_samples)} total). "
                            "Reduce DEV; CB and Copa are capped by their launcher at DEV=100."
                        )
                    # Dev samples
                    dev_samples = train_samples[-args.num_dev:] 
                    train_samples = train_samples[:-args.num_dev]
                else:
                    dev_samples = None

                val_samples = dev_samples if dev_samples is not None else eval_samples
                framework.test_samples = eval_samples

                # Training
                logger.info(f"Starting training for set {train_set_id}...")
                train_start_time = time.time()
                
                framework.train(train_samples, val_samples)
                
                train_end_time = time.time()
                train_duration = train_end_time - train_start_time
                logger.info(f"Training for set {train_set_id} finished in {train_duration:.2f} seconds.")

                if not args.no_eval:
                    trainer = getattr(framework, "trainer", None)
                    metrics = {}

                    # Load validation-best checkpoint (if any), then report only that model.
                    best_ckpt = trainer.state.best_model_checkpoint if trainer is not None else None
                    if best_ckpt is not None:
                        best_step = _resolve_best_checkpoint_step(trainer, best_ckpt)
                        last_step = trainer.state.global_step if trainer is not None else None
                        target_model = trainer.model if trainer is not None else framework.model
                        if (
                            best_step is not None
                            and last_step is not None
                            and int(best_step) == int(last_step)
                        ):
                            logger.info(
                                f"Last step {last_step} is already best; skip reload from {best_ckpt}"
                            )
                        else:
                            logger.info(f"Loading best checkpoint for final eval: {best_ckpt}")
                            # expect_change=False: if model already matches disk (e.g. last==best
                            # but step metadata missing, or trainer already reloaded), do not fail.
                            _load_checkpoint_weights(target_model, best_ckpt, expect_change=False)
                        framework.model = target_model
                        if trainer is not None:
                            trainer.model = target_model
                        metrics["best_checkpoint_step"] = best_step
                    else:
                        logger.warning("No best checkpoint found; evaluating the final training state.")
                        if trainer is not None:
                            metrics["best_checkpoint_step"] = trainer.state.global_step

                    # Test-set metrics (no prefix) + optional validation metrics (dev_*).
                    metrics.update(_collect_split_metrics(framework, eval_samples, ""))
                    if dev_samples is not None:
                        metrics.update(_collect_split_metrics(framework, val_samples, "dev_"))

                    if trainer is not None and getattr(trainer, "gpu_memory_max_allocated_mb", None) is not None:
                        metrics["gpu_memory_max_allocated_mb"] = trainer.gpu_memory_max_allocated_mb
                    elif torch.cuda.is_available():
                        metrics["gpu_memory_max_allocated_mb"] = round(
                            torch.cuda.max_memory_allocated() / (1024 ** 2), 2
                        )
            else:
                # Zero-shot / in-context learning.  When ``num_dev`` is
                # supplied together with a fixed train-set seed, the sampled
                # training array contains ``num_train`` examples followed by
                # the held-out development examples.  Supporting this split
                # in trainer-none mode lets us finalize an interrupted
                # training run from ``checkpoint-best`` without pretending
                # that its test score is also its dev score.
                if args.num_dev is not None:
                    if args.num_dev <= 0 or len(train_samples) < args.num_dev:
                        raise ValueError(
                            f"num_dev={args.num_dev} is incompatible with the "
                            f"sampled set of {len(train_samples)} examples."
                        )
                    dev_samples = train_samples[-args.num_dev:]
                    metrics = framework.evaluate([], eval_samples)
                    metrics.update(_collect_split_metrics(framework, dev_samples, "dev_"))
                else:
                    metrics = framework.evaluate(train_samples, eval_samples)

            if not args.no_eval:
                if train_duration is not None:
                    metrics["total_wall_clock_time_seconds"] = round(train_duration, 2)
                    metrics["total_wall_clock_time"] = _format_duration(train_duration)
                logger.info("===== Train set %d =====" % train_set_seed)
                logger.info(metrics)
                maybe_log_to_wandb(prefix_metric_keys(_to_wandb_safe_metrics(metrics), "final/"))
                if args.local_rank <= 0:
                    import os
                    result_dir = os.path.join("result_uv", args.trainer)
                    os.makedirs(result_dir, exist_ok=True)
                    result_path = os.path.join(result_dir, result_file_tag(args) + f"-trainset{train_set_id}.json" if args.result_file is None else args.result_file)
                    write_metrics_to_file(metrics, result_path)

    else:
        # For each eval sample, there is a training set. no training is allowed
        # This is for in-context learning (ICL)
        assert args.trainer == "none"
        if args.num_eval is not None:
            eval_samples = task.sample_subset(data_split="valid", seed=0, num=args.num_eval)
        else:
            eval_samples = task.valid_samples

        metrics = framework.evaluate(train_sets, eval_samples, one_train_set_per_eval_sample=True)
        logger.info(metrics)
        maybe_log_to_wandb(prefix_metric_keys(_to_wandb_safe_metrics(metrics), "final/"))

    # --- End Timer & Print Total Time ---
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time

    logger.info("=" * 30)
    logger.info(f"Total Process Finished.")
    if train_duration is not None:
        logger.info(
            f"Training Wall Clock Time: {_format_duration(train_duration)} "
            f"({train_duration:.2f} seconds)"
        )
    logger.info(
        f"Total Process Wall Clock Time: {_format_duration(total_duration)} "
        f"({total_duration:.2f} seconds total)"
    )
    logger.info("=" * 30)
    if result_path is not None and metrics:
        # JSON wall-clock fields already store training-only time when available.
        write_metrics_to_file(metrics, result_path)

if __name__ == "__main__": 
    main()
