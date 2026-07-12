"""
File: losses.py

Description:
    Loss functions and multi-task loss combiner with Kendall weighting.

Purpose:
    Encapsulate all loss computation so the training loop only calls
    total_loss(outputs, batch) → (loss, loss_dict, raw_loss_dict, kendall_weights).

Usage:
    from src.training.losses import LossCombiner, AsymmetricLoss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LossCombiner(nn.Module):
    """Multi-task loss combiner with Kendall uncertainty weighting.

    Supports three phases:
    - calibration: accumulates per-task loss means
    - mean: normalizes each loss by its running mean
    - kendall: learns task weights via softplus parameterization
    """

    def __init__(self, losses: dict[str, nn.Module]) -> None:
        super().__init__()

        self.losses = losses
        self.loss_names = list(losses.keys())
        self.loss_means: dict[str, float] = {name: 1.0 for name in self.loss_names}

        # Kendall parameters
        self.raw_weights = nn.Parameter(torch.ones(len(losses)))

        # phase control
        self.phase = "calibration"

        # running stats (ONLY for init calibration)
        self.loss_sum: dict[str, float] = {name: 0.0 for name in self.loss_names}
        self.loss_counts: dict[str, int] = {name: 0 for name in self.loss_names}

        self.tasks: dict[str, dict] = {
            "food_type": {"target": "food_type", "type": "cls"},
            "ingredients": {"target": "ingredients", "type": "multi"},
            "portion_presence": {"target": "portion_presence", "type": "multi"},
            "cooking_method": {"target": "cooking_method", "type": "multi"},
            "dish_name": {"target": "dish_name", "type": "cls"},
            "portion_weight": {"target": "portion_weight", "type": "reg"},
            "calories": {"target": "calories_kcal", "type": "reg"},
            "fats": {"target": "fat_g", "type": "reg"},
            "carbohydrates": {"target": "carbohydrate_g", "type": "reg"},
            "proteins": {"target": "protein_g", "type": "reg"},
            "binary": {
                "target": ["food_prob", "camera_or_phone_prob"],
                "type": "multi_bin",
            },
        }

    def reset_means(self) -> None:
        """Reset running loss sums and counts."""
        self.loss_sum = {name: 0.0 for name in self.loss_names}
        self.loss_counts = {name: 0 for name in self.loss_names}

    def set_mean_phase(self) -> None:
        """Freeze running means and switch to mean-normalized phase."""
        with torch.no_grad():
            for name in self.loss_names:
                self.loss_means[name] = self.loss_sum[name] / max(self.loss_counts[name], 1)
        self.phase = "mean"

    def set_kendall_phase(self) -> None:
        """Initialize Kendall weights and switch to learned weighting phase."""
        self.raw_weights.data.fill_(0.5413248546)
        self.phase = "kendall"

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, float]]:
        """Compute weighted total loss across all tasks.

        Returns:
            total_loss: scalar weighted loss
            loss_dict: detached weighted losses per task
            raw_loss_dict: detached unweighted losses per task
            kendall_weights: current softplus weights per task
        """
        total_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        loss_dict: dict[str, torch.Tensor] = {}
        raw_loss_dict: dict[str, torch.Tensor] = {}

        for i, name in enumerate(self.loss_names):
            task = self.tasks[name]

            # resolve target
            if isinstance(task["target"], list):
                target = torch.stack([targets[t] for t in task["target"]], dim=1)
            else:
                target = targets[task["target"]]

            output = outputs[f"{name}_logits"]

            if task["type"] == "reg":
                target = target.float()
                if target.ndim == 1:
                    target = target.unsqueeze(1)

            if task["target"] == "portion_weight":
                output = output * targets["portion_presence"]

            base_loss = self.losses[name](output, target)

            # calibration: just accumulate stats
            if self.phase == "calibration":
                self.loss_sum[name] += base_loss.detach().item()
                self.loss_counts[name] += 1
                weighted_loss = base_loss
            # mean: normalize by running mean
            elif self.phase == "mean":
                weighted_loss = base_loss / self.loss_means[name]
            # kendall: learn task weights
            else:
                weight = F.softplus(self.raw_weights[i]) + 1e-8
                normalized_loss = base_loss / self.loss_means[name]
                if task["type"] == "reg":
                    weighted_loss = normalized_loss / weight.pow(2) + 0.5 * torch.log(weight)
                else:
                    weighted_loss = normalized_loss / weight.pow(2) + torch.log(weight)

            loss_dict[name] = weighted_loss.detach()
            raw_loss_dict[name] = base_loss.detach()
            total_loss = total_loss + weighted_loss

        kendall_weights = {
            name: F.softplus(self.raw_weights[i]).item()
            for i, name in enumerate(self.loss_names)
        }

        return total_loss, loss_dict, raw_loss_dict, kendall_weights


class AsymmetricLoss(nn.Module):
    """Asymmetric loss for multi-label classification.

    Down-weights easy negatives more aggressively than easy positives.
    """

    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        gamma_neg: float = 4,
        gamma_pos: float = 1,
        clip: float = 0.05,
    ) -> None:
        super().__init__()
        self.class_weights = class_weights
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)

        pos = targets
        neg = 1 - targets

        xs_pos = probs.clamp(min=1e-8, max=1 - 1e-8)
        xs_neg = (1 - probs).clamp(min=1e-8, max=1 - 1e-8)

        loss_pos = pos * torch.log(xs_pos)
        loss_neg = neg * torch.log(xs_neg)

        loss_pos *= (1 - xs_pos).pow(self.gamma_pos)
        loss_neg *= xs_pos.pow(self.gamma_neg)

        loss = -(loss_pos + loss_neg)

        if self.class_weights is not None:
            loss = loss * self.class_weights.view(1, -1)

        return loss.mean()
