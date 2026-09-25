"""htdemucs_ft v4 (Meta, per-source fine-tuned) - AUTO. ~4x slower than htdemucs."""
from separators._demucs_base import DemucsSeparator


def build():
    return DemucsSeparator(name="htdemucs_ft", model="htdemucs_ft")
