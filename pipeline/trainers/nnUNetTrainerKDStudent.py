"""Shared KD trainer for the student-plan C1/C2/C3 experiments."""
from __future__ import annotations

import json
import os
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import autocast

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

from .soft_cldice import soft_cldice_loss


def checkpoint_weights_for_model(checkpoint, model):
    """Map DDP/torch.compile checkpoint keys to the active model's key format."""
    weights = checkpoint.get("network_weights", checkpoint)
    target_keys = set(model.state_dict())
    matched = {}
    for key, value in weights.items():
        plain_key = key.removeprefix("module.").removeprefix("_orig_mod.")
        candidates = (key, plain_key, f"_orig_mod.{plain_key}")
        destination = next((candidate for candidate in candidates if candidate in target_keys), None)
        if destination is not None:
            matched[destination] = value
    return matched


class nnUNetTrainerKDStudent(nnUNetTrainer):
    DEFAULT_ALPHA = 0.7
    DEFAULT_USE_CLDICE = False

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        self.num_epochs = int(os.environ.get("KD_NUM_EPOCHS", 5))
        self.kd_alpha = float(os.environ.get("KD_ALPHA", self.DEFAULT_ALPHA))
        self.kd_temperature = float(os.environ.get("KD_TEMPERATURE", 3.0))
        self.kd_teacher_checkpoint = os.environ.get("KD_TEACHER_CHECKPOINT")
        self.kd_teacher_plans_path = os.environ.get("KD_TEACHER_PLANS")
        self.kd_student_checkpoint = os.environ.get("KD_STUDENT_CHECKPOINT")
        self.kd_warm_start = os.environ.get("KD_WARM_START", "1") == "1"
        self.kd_use_cldice = os.environ.get("KD_USE_CLDICE", "1" if self.DEFAULT_USE_CLDICE else "0") == "1"
        self.kd_cldice_weight = float(os.environ.get("KD_CLDICE_WEIGHT", 0.3))
        self.teacher_network = None

    @staticmethod
    def build_network_architecture(plans_manager, configuration_manager, num_input_channels,
                                   num_output_channels, enable_deep_supervision=True):
        # ``nnUNetPlansStudent`` already contains the narrowed feature widths.
        # Do not multiply them again here.
        return get_network_from_plans(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            allow_init=True,
            deep_supervision=enable_deep_supervision,
        )

    def _build_teacher_network(self):
        if not self.kd_teacher_checkpoint or not Path(self.kd_teacher_checkpoint).is_file():
            raise RuntimeError("KD_TEACHER_CHECKPOINT must point to teacher checkpoint_final.pth.")
        if not self.kd_teacher_plans_path or not Path(self.kd_teacher_plans_path).is_file():
            raise RuntimeError("KD_TEACHER_PLANS must point to the teacher plans.json.")
        with open(self.kd_teacher_plans_path) as handle:
            plans_manager = PlansManager(json.load(handle))
        configuration = plans_manager.get_configuration(self.configuration_name)
        teacher = nnUNetTrainer.build_network_architecture(
            plans_manager, configuration, self.num_input_channels,
            self.label_manager.num_segmentation_heads, enable_deep_supervision=True,
        ).to(self.device)
        checkpoint = torch.load(self.kd_teacher_checkpoint, map_location=self.device, weights_only=False)
        weights = checkpoint_weights_for_model(checkpoint, teacher)
        teacher.load_state_dict(weights, strict=True)
        teacher.decoder.deep_supervision = False
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        return teacher

    def _maybe_warm_start(self):
        checkpoint_file = Path(self.kd_student_checkpoint) if self.kd_student_checkpoint else None
        latest = Path(self.output_folder) / "checkpoint_latest.pth"
        if not self.kd_warm_start or checkpoint_file is None or not checkpoint_file.is_file() or latest.exists():
            return
        checkpoint = torch.load(checkpoint_file, map_location=self.device, weights_only=False)
        weights = checkpoint_weights_for_model(checkpoint, self.network)
        try:
            self.network.load_state_dict(weights, strict=True)
        except RuntimeError as error:
            self.print_to_log_file(f"[KD] baseline checkpoint is incompatible; using random student weights: {error}")
            return
        self.print_to_log_file(f"[KD] warm-started compatible student weights from {checkpoint_file}")

    def on_train_start(self):
        super().on_train_start()
        self._maybe_warm_start()
        self.teacher_network = self._build_teacher_network()
        self.print_to_log_file(f"[KD] frozen teacher={self.kd_teacher_checkpoint}; alpha={self.kd_alpha}; temperature={self.kd_temperature}")

    def train_step(self, batch):
        data, target = batch["data"].to(self.device, non_blocking=True), batch["target"]
        target = [item.to(self.device, non_blocking=True) for item in target] if isinstance(target, list) else target.to(self.device, non_blocking=True)
        self.optimizer.zero_grad(set_to_none=True)
        context = autocast(self.device.type, enabled=True) if self.device.type == "cuda" else nullcontext()
        with context:
            student_output = self.network(data)
            supervised_loss = self.loss(student_output, target)
            student_logits = student_output[0] if isinstance(student_output, (list, tuple)) else student_output
            with torch.no_grad():
                teacher_logits = self.teacher_network(data)
                teacher_logits = teacher_logits[0] if isinstance(teacher_logits, (list, tuple)) else teacher_logits
                if teacher_logits.shape[2:] != student_logits.shape[2:]:
                    mode = "trilinear" if student_logits.ndim == 5 else "bilinear"
                    teacher_logits = F.interpolate(teacher_logits, size=student_logits.shape[2:], mode=mode, align_corners=False)
            temperature = self.kd_temperature
            distill_loss = F.kl_div(F.log_softmax(student_logits.float() / temperature, dim=1), F.softmax(teacher_logits.float() / temperature, dim=1), reduction="batchmean") * temperature**2
            loss = self.kd_alpha * supervised_loss + (1 - self.kd_alpha) * distill_loss
            cldice_loss = None
            if self.kd_use_cldice:
                labels = target[0] if isinstance(target, list) else target
                one_hot = torch.zeros_like(student_logits).scatter_(1, labels.long(), 1)
                cldice_loss = soft_cldice_loss(F.softmax(student_logits.float(), dim=1), one_hot)
                loss = loss + self.kd_cldice_weight * cldice_loss
        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        output = {"loss": loss.detach().cpu().numpy(), "supervised_loss": supervised_loss.detach().cpu().numpy(), "distill_loss": distill_loss.detach().cpu().numpy()}
        if cldice_loss is not None:
            output["cldice_loss"] = cldice_loss.detach().cpu().numpy()
        return output
