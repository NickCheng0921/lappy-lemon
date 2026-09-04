"""htdemucs core: ONNX session wrapped in the torch STFT front/back end."""

import sys
from pathlib import Path

from gains import SOURCES
from util import log


class Separator:
    """Wraps the torch front/back end around the ONNX core."""

    def __init__(self, model_path, repo, threads=1, lean=True):
        sys.path.insert(0, str(Path(repo) / "vendor"))
        sys.path.insert(0, str(Path(repo) / "profile"))
        import onnxruntime as ort
        import torch as th
        from demucs.htdemucs import HTDemucs
        from onnx_export import front, reconstruct

        self.th, self.front, self.reconstruct = th, front, reconstruct

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if lean:
            so.enable_cpu_mem_arena = False
            so.enable_mem_pattern = False
        self.sess = ort.InferenceSession(
            model_path, so, providers=["CPUExecutionProvider"]
        )

        # The input shape tells us the window the graph was exported at. Trust
        # the model, never a CLI flag, or the STFT frames will not line up.
        shp = {i.name: i.shape for i in self.sess.get_inputs()}
        self.win = int(shp["mix_t"][2])

        # HTDemucs instance used ONLY for its STFT/mask/iSTFT helpers, which
        # have no learned parameters. Built at the student's width so segment
        # and nfft match the exported graph; the weights are never used.
        from fractions import Fraction

        self.m = HTDemucs(
            sources=SOURCES,
            channels=24,
            bottom_channels=256,
            t_layers=4,
            segment=Fraction(self.win, 44100).limit_denominator(100000),
        ).eval()
        self.sr = self.m.samplerate
        self.ch = self.m.audio_channels
        log(
            f"model {Path(model_path).name}: window {self.win} samples "
            f"({self.win/self.sr:.2f}s) @ {self.sr}Hz"
        )

    def separate(self, chunk):
        """chunk: float32 [ch, win] -> stems float32 [4, ch, win]"""
        th = self.th
        with th.no_grad():
            mix = th.from_numpy(chunk[None])  # [1, ch, win]
            ctx = self.front(self.m, mix)
            x, xt = self.sess.run(
                None, {"mag": ctx["x"].numpy(), "mix_t": ctx["xt"].numpy()}
            )
            out = self.reconstruct(self.m, th.from_numpy(x), th.from_numpy(xt), ctx)
        return out[0].numpy()  # [4, ch, win]
