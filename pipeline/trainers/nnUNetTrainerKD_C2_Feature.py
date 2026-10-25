"""C2: supervised feature-level distillation."""
from __future__ import annotations

import os
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast

from .nnUNetTrainerKDStudent import nnUNetTrainerKDStudent


class nnUNetTrainerKD_C2_Feature(nnUNetTrainerKDStudent):
    DEFAULT_ALPHA = 0.7
    DEFAULT_USE_CLDICE = False

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        self.kd_feature_weight = float(os.environ.get("KD_FEATURE_WEIGHT", 0.1))
        self.feature_projector = None

    def on_train_start(self):
        super().on_train_start()
        if self.feature_projector is None:
            patch_size = self.configuration_manager.patch_size
            with torch.no_grad():
                sample = torch.zeros(1, self.num_input_channels, *patch_size, device=self.device)
                student_channels = self.network.encoder(sample)[-1].shape[1]
                teacher_channels = self.teacher_network.encoder(sample)[-1].shape[1]
            convolution = nn.Conv3d if len(patch_size) == 3 else nn.Conv2d
            self.feature_projector = convolution(student_channels, teacher_channels, 1).to(self.device)
            self.optimizer.add_param_group({"params": self.feature_projector.parameters()})
        self.print_to_log_file(f"[KD-C2] feature_weight={self.kd_feature_weight}")

    def train_step(self, batch: dict) -> dict:
        data, target = batch["data"].to(self.device, non_blocking=True), batch["target"]
        target = [item.to(self.device, non_blocking=True) for item in target] if isinstance(target, list) else target.to(self.device, non_blocking=True)
        self.optimizer.zero_grad(set_to_none=True)
        context = autocast(self.device.type, enabled=True) if self.device.type == "cuda" else nullcontext()
        with context:
            student_skips = self.network.encoder(data)
            student_output = self.network.decoder(student_skips)
            supervised_loss = self.loss(student_output, target)
            with torch.no_grad():
                teacher_features = self.teacher_network.encoder(data)[-1]
            projected = self.feature_projector(student_skips[-1].float())
            if projected.shape[2:] != teacher_features.shape[2:]:
                mode = "trilinear" if projected.ndim == 5 else "bilinear"
                projected = F.interpolate(projected, size=teacher_features.shape[2:], mode=mode, align_corners=False)
            feature_loss = F.mse_loss(projected, teacher_features.float())
            loss = self.kd_alpha * supervised_loss + self.kd_feature_weight * feature_loss
        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
        return {"loss": loss.detach().cpu().numpy(), "supervised_loss": supervised_loss.detach().cpu().numpy(), "feature_loss": feature_loss.detach().cpu().numpy()}
