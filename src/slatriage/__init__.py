"""SlaTriage: multi-agent LoRA adjudication of Slither alerts."""

__version__ = "0.1.0"

from .council import run_config
from .families import assign_family
from .schema import Alert, ExpertDecision, JudgeOutput

__all__ = [
    "__version__",
    "Alert",
    "ExpertDecision",
    "JudgeOutput",
    "run_config",
    "assign_family",
]
