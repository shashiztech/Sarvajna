"""
Sarvanjna - Multimodal AI Platform

A production-grade platform for text, image, video, and music generation.
"""

__version__ = "0.1.0"
__author__ = "Research AI Team"

from sarvanjna.core.config import Config
from sarvanjna.core.registry import ModelRegistry

__all__ = ["Config", "ModelRegistry", "__version__"]
