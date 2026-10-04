import json
import logging
import os
from dataclasses import asdict, is_dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


def _canonicalize_name(name: str) -> str:
    prefixes = ("module.", "_fsdp_wrapped_module.")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if name.startswith(prefix):
                name = name[len(prefix):]
                changed = True
    return name


def _sanitize_filename_part(value: str) -> str:
    safe = []
    for ch in value:
        if ch.isalnum() or ch in ("-", "_", "."):
            safe.append(ch)
        else:
            safe.append("-")
    return "".join(safe).strip("-") or "unknown"


def _reshape_to_matrix(tensor: torch.Tensor) -> Optional[torch.Tensor]:
    if tensor.ndim < 2:
        return None
    if tensor.ndim == 2:
        return tensor
    return tensor.reshape(tensor.shape[0], -1)


def _candidate_layers(named_parameters: Iterable[Tuple[str, torch.nn.Parameter]]) -> List[Tuple[str, torch.nn.Parameter]]:
    preferred = []
    fallback = []
    for name, param in named_parameters:
        if not param.requires_grad:
            continue
        if _reshape_to_matrix(param.detach()) is None:
            continue
        canonical_name = _canonicalize_name(name)
        fallback.append((canonical_name, param))
        lowered = canonical_name.lower()
        if any(token in lowered for token in ("bias", "layernorm", "layer_norm", "norm", "embed", "embedding")):
            continue
        preferred.append((canonical_name, param))
    return preferred if preferred else fallback


def select_layers(
    named_parameters: Iterable[Tuple[str, torch.nn.Parameter]],
    requested_layers: str,
    max_layers: int,
) -> List[str]:
    candidates = list(_candidate_layers(named_parameters))
    if requested_layers:
        patterns = [item.strip() for item in requested_layers.split(",") if item.strip()]
        selected = []
        for name, _ in candidates:
            if any(pattern in name for pattern in patterns):
                selected.append(name)
        return selected[:max_layers]

    ranked = sorted(candidates, key=lambda item: item[1].numel(), reverse=True)
    return [name for name, _ in ranked[:max_layers]]


class GradientSpectrumRecorder:
    def __init__(self, args, model, method_name: str):
        self.enabled = bool(getattr(args, "save_gradient_spectra", False))
        self.method_name = method_name
        self.max_values = max(1, int(getattr(args, "svd_max_values", 256)))
        self.interval = max(1, int(getattr(args, "svd_save_interval", 100)))
        self.output_dir = self._resolve_output_dir(args)
        file_stem = self._resolve_file_stem(args)
        self.record_path = os.path.join(self.output_dir, f"{file_stem}.jsonl")
        self.meta_path = os.path.join(self.output_dir, f"{file_stem}_meta.json")
        self.selected_layers = []
        self._selected_layer_set = set()
        self._args = args

        if not self.enabled:
            return

        self.selected_layers = select_layers(
            model.named_parameters(),
            getattr(args, "svd_layers", ""),
            max(1, int(getattr(args, "svd_max_layers", 4))),
        )
        self._selected_layer_set = set(self.selected_layers)

        os.makedirs(self.output_dir, exist_ok=True)
        for path in (self.record_path, self.meta_path):
            if os.path.exists(path):
                os.remove(path)
        self._write_metadata(model)
        logger.info("Saving %s gradient spectra to %s", self.method_name, self.output_dir)

    def _resolve_output_dir(self, args) -> str:
        requested_dir = getattr(args, "svd_output_dir", None)
        if requested_dir:
            return requested_dir

        base_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base_dir, "svd_spectra")

    def _resolve_file_stem(self, args) -> str:
        task_name = _sanitize_filename_part(getattr(args, "task_name", "task"))
        model_name = _sanitize_filename_part(str(getattr(args, "model_name", "model")).split("/")[-1])
        return f"{task_name}-{model_name}-{self.method_name}"

    def _write_metadata(self, model) -> None:
        arg_dict = {}
        if is_dataclass(self._args):
            for key, value in asdict(self._args).items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    arg_dict[key] = value
                else:
                    arg_dict[key] = str(value)
        metadata = {
            "method": self.method_name,
            "selected_layers": self.selected_layers,
            "interval": self.interval,
            "max_values": self.max_values,
            "args": arg_dict,
        }
        with open(self.meta_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

    def should_capture(self, step: int) -> bool:
        return self.enabled and bool(self.selected_layers) and step > 0 and step % self.interval == 0

    def is_selected(self, name: str) -> bool:
        return _canonicalize_name(name) in self._selected_layer_set

    def capture_from_named_tensors(
        self,
        step: int,
        source: str,
        named_tensors: Dict[str, torch.Tensor],
        extra: Optional[Dict] = None,
    ) -> None:
        if not self.should_capture(step):
            return

        spectra = {}
        shapes = {}
        for original_name, tensor in named_tensors.items():
            canonical_name = _canonicalize_name(original_name)
            if canonical_name not in self._selected_layer_set:
                continue
            matrix = _reshape_to_matrix(tensor.detach())
            if matrix is None:
                continue
            matrix = matrix.float().cpu()
            singular_values = torch.linalg.svdvals(matrix)[: self.max_values]
            spectra[canonical_name] = singular_values.tolist()
            shapes[canonical_name] = list(matrix.shape)

        if not spectra:
            return

        payload = {
            "step": int(step),
            "source": source,
            "method": self.method_name,
            "layers": spectra,
            "shapes": shapes,
        }
        if extra:
            payload["extra"] = extra

        with open(self.record_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")
