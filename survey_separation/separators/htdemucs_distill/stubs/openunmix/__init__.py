# Stub openunmix: shadow the real package so demucs.hdemucs can import
# `wiener` without pulling torchaudio (its prebuilt .so is ABI-broken here).
# htdemucs runs wiener_iters=0, so wiener is never called. See CLAUDE.md.
