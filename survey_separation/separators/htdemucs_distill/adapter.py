"""htdemucs_distill - the distilled student ONNX (student_final_slim.onnx, ~8.8M
params) run through demucs' apply_model with the ORT core. STFT front/back stay in
torch (onnx_export.OrtCoreModel); normalization + overlap-add mirror the demucs
CLI so results are comparable to the stock htdemucs row.
"""
from __future__ import annotations
import os
from pathlib import Path

from benchmark.core.separator import Separator
from benchmark.core.stems import StemSet, CANONICAL

REPO = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = REPO / "pi_optimize_demucs" / "models" / "student_final_slim.onnx"


class DistillSeparator(Separator):
    produces = set(CANONICAL)

    def __init__(self, model_path):
        self.name = "htdemucs_distill"
        self.model_path = str(model_path)

    def resolve(self, track, workdir):
        out_dir = Path(workdir) / self.name / track.id
        paths = {s: out_dir / f"{s}.wav" for s in CANONICAL}
        if not all(p.exists() for p in paths.values()):
            out_dir.mkdir(parents=True, exist_ok=True)
            self._separate(track.mixture, paths)
        return StemSet(track.id, paths,
                       meta={"tool": self.name, "model": Path(self.model_path).name})

    def _separate(self, mixture, paths):
        import sys
        import numpy as np
        import soundfile as sf
        import torch as th

        # Stub openunmix first so demucs.hdemucs imports `wiener` without pulling
        # torchaudio (broken .so). htdemucs runs wiener_iters=0.
        sys.path.insert(0, str(Path(__file__).parent / "stubs"))
        sys.path.insert(0, str(REPO / "profile"))
        sys.path.insert(0, str(REPO / "vendor"))
        import onnxruntime as ort
        from onnx_export import OrtCoreModel, get_htdemucs
        from demucs.apply import apply_model

        # Teacher-config htdemucs, used ONLY for its (parameter-free) STFT/mask/
        # iSTFT helpers and config; the actual weights come from the student ONNX.
        m = get_htdemucs("htdemucs")
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(os.environ.get("DISTILL_THREADS", "0")) or (os.cpu_count() or 4)
        sess = ort.InferenceSession(self.model_path, so, providers=["CPUExecutionProvider"])
        model = OrtCoreModel(m, sess).eval()

        wav, sr = sf.read(str(mixture), dtype="float32", always_2d=True)  # [T, C]
        wav = th.from_numpy(wav.T)                                        # [C, T]
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)

        # Global normalization around apply_model, exactly as demucs api.py.
        ref = wav.mean(0)
        wav = (wav - ref.mean()) / (ref.std() + 1e-8)
        with th.no_grad():
            out = apply_model(model, wav[None], shifts=1, split=True, overlap=0.25,
                              device="cpu", num_workers=0, progress=False)[0]  # [S, C, T]
        out = out * (ref.std() + 1e-8) + ref.mean()

        for i, s in enumerate(m.sources):        # ["drums","bass","other","vocals"]
            sf.write(str(paths[s]), out[i].numpy().T, 44100, subtype="PCM_16")


def build(model_path=None):
    return DistillSeparator(model_path or os.environ.get("DISTILL_MODEL", str(DEFAULT_MODEL)))
