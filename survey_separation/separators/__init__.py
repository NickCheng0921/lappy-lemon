"""Adapter registry. Each tool lives in its own subpackage exposing build()."""
from __future__ import annotations


def default_separators(spleeter_python: str = "python"):
    from separators.htdemucs.adapter import build as htdemucs
    from separators.htdemucs_ft.adapter import build as htdemucs_ft
    from separators.spleeter.adapter import build as spleeter
    from separators.flstudio.adapter import build as flstudio
    from separators.virtualdj.adapter import build as virtualdj
    from separators.jbl_bandbox.adapter import build as jbl_bandbox
    return [
        htdemucs(),
        htdemucs_ft(),
        spleeter(spleeter_python),
        flstudio(),
        virtualdj(),
        jbl_bandbox(),
    ]
