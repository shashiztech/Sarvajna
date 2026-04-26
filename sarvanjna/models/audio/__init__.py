"""
Audio models for music generation.

Implements neural audio codec and music generation.
"""

from .audio_codec import EnCodec, EnCodecConfig
from .music_generator import MusicGen, MusicGenConfig

__all__ = [
    'EnCodec',
    'EnCodecConfig',
    'MusicGen',
    'MusicGenConfig',
]
