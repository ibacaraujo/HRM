import torch
from torch import nn
from typing import Dict, Tuple

from models.layers import CastedLinear
from models.hrm.hrm_act_v1 import (
    HierarchicalReasoningModel_ACTV1,
    HierarchicalReasoningModel_ACTV1Carry,
)


class HierarchicalReasoningModel_ACTV1_Classifier(nn.Module):
    """HRM wrapper for image classification."""

    def __init__(self, config_dict: dict, num_classes: int):
        super().__init__()
        self.hrm = HierarchicalReasoningModel_ACTV1(config_dict)
        self.num_classes = num_classes
        self.classifier = CastedLinear(
            self.hrm.config.hidden_size, num_classes, bias=True
        )

    @property
    def puzzle_emb(self):
        return self.hrm.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        return self.hrm.initial_carry(batch)

    def forward(
        self,
        carry: HierarchicalReasoningModel_ACTV1Carry,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[HierarchicalReasoningModel_ACTV1Carry, Dict[str, torch.Tensor]]:
        carry, outputs = self.hrm(carry=carry, batch=batch)
        hidden = carry.inner_carry.z_H[:, self.hrm.inner.puzzle_emb_len :, :]
        outputs["cls_logits"] = self.classifier(hidden.mean(dim=1))
        return carry, outputs
