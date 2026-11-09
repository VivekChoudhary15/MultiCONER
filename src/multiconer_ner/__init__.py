"""MultiCoNER multilingual NER training pipeline."""

from .config import Config
from .pipeline import run_pipeline

__all__ = ["Config", "run_pipeline"]
