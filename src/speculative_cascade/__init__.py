"""Cascading Speculative Acceleration for LLM Inference.

This package implements a 3-stage speculative decoding system optimized
for Google Cloud TPU v5e Pods.
"""

__version__ = "0.1.0"
__author__ = "Marco Durán Cabobianco"
__email__ = "marco@anachroni.co"

from speculative_cascade.core.cascade import CascadeInference
from speculative_cascade.core.verification import DistributedVerifier
from speculative_cascade.models.tiny import TinyModel
from speculative_cascade.models.draft import DraftModel
from speculative_cascade.models.target import TargetModel

__all__ = [
    "CascadeInference",
    "DistributedVerifier",
    "TinyModel",
    "DraftModel",
    "TargetModel",
]
