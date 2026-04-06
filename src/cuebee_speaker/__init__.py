"""CueBee multi-tenant speaker diarization serving."""

from .config import AppConfig
from .domain import AudioChunk, SpeakerSegment, STTSegment

__all__ = ["AppConfig", "AudioChunk", "SpeakerSegment", "STTSegment"]
__version__ = "0.1.0"

