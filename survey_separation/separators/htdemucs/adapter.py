"""htdemucs v4 (Meta) - AUTO. Stock weights via the demucs CLI."""
from separators._demucs_base import DemucsSeparator


def build():
    return DemucsSeparator(name="htdemucs", model="htdemucs")
