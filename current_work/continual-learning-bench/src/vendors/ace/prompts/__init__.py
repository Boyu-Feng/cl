"""Vendored ACE prompts."""

from .curator import CURATOR_PROMPT, CURATOR_PROMPT_NO_GT
from .generator import GENERATOR_PROMPT
from .reflector import REFLECTOR_PROMPT, REFLECTOR_PROMPT_NO_GT

__all__ = [
    "GENERATOR_PROMPT",
    "REFLECTOR_PROMPT",
    "REFLECTOR_PROMPT_NO_GT",
    "CURATOR_PROMPT",
    "CURATOR_PROMPT_NO_GT",
]
