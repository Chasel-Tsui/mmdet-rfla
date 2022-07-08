import torch

from ..builder import BBOX_ASSIGNERS
from ..iou_calculators import build_iou_calculator
from .assign_result import AssignResult
from .base_assigner import BaseAssigner

@BBOX_ASSIGNERS.register_module()
class MaxIoUAssigner(BaseAssigner):