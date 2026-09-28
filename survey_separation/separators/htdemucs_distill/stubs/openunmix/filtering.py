"""Stub for openunmix.filtering.wiener. Never called (htdemucs wiener_iters=0);
exists only so `from openunmix.filtering import wiener` succeeds without importing
torchaudio."""


def wiener(*args, **kwargs):
    raise NotImplementedError("wiener stub: htdemucs runs with wiener_iters=0")
