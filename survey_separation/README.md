# survey_separation

Compare performance of stem splitting tools available (2026)

## Results

MUSDB18 test set (50 tracks), BSS Eval v4, 1.0 s window.

SDR (Signal-to-Distortion Ratio) is the class of metric used to judge separation quality, implementation from BSS Eval v4. SDR represents the ratio between the true stem track and the amount of error the
separation model adds.

$$\mathrm{SDR} = 10\,\log_{10}\frac{\lVert s_{\text{target}}\rVert^2}{\lVert e_{\text{spat}} + e_{\text{interf}} + e_{\text{artif}}\rVert^2}\quad[\text{dB}]$$

<sub>  [signal vs. spatial + interference + artifact error, BSS Eval v4](https://github.com/sigsep/sigsep-mus-eval/blob/master/museval/metrics.py)</sub>

For example, a 3 dB improvement equates to ~2× cleaner separation: $10^{3/10} \approx 2$.

**4stem — SDR (dB)**
| separator   | bass (med / mean) | drums (med / mean) | other (med / mean) | vocals (med / mean) |
|:------------|:-----------------:|:------------------:|:------------------:|:-------------------:|
| htdemucs    |   10.22 / 9.11    |    9.98 / 10.20    |    6.48 / 6.11     |    8.66 / 7.78      |
| htdemucs_ft |   10.34 / 9.37    |   10.22 / 10.41    |    6.37 / 6.01     |    8.79 / 8.11      |
| virtualdj   |    8.98 / 8.00    |    9.61 / 10.06    |    6.20 / 5.76     |   10.06 / 9.53      |
| htdemucs_distill |    5.65 / 4.63    |    5.70 / 6.15     |    4.03 / 3.66     |    5.76 / 5.31      |

<sub>htdemucs_distill is a 8.8M distillation of the 42M htdemucs model targetting a rpi 5

**[Published Values](https://huggingface.co/datasets/StemSplitio/stem-separation-benchmark-2026) - 4stem (median SDR, dB)**
| model_id         |  bass |  drums | other | vocals |
|:-----------------|------:|-------:|------:|-------:|
| htdemucs         |  9.78 |  10.01 |  6.42 |   8.53 |
| htdemucs_ft      | 10.38 |  10.11 |  6.34 |   9.19 |

Published figures on lossless MUSDB18-HQ. Collected metrics are on compressed mp4 MUSDB18 (decoded from `.stem.mp4`) with a 1.0 s window, so it sits a touch lower.

## Ingestion, by tool
| Tool           | How stems arrive                          |
|----------------|-------------------------------------------|
| htdemucs       | `demucs` CLI (auto)                       |
| htdemucs_ft    | `demucs` CLI (auto)                       |
| spleeter       | `spleeter` CLI, separate venv (auto)      |
| flstudio       | GUI export -> `flstudio/inbox/<id>/`      |
| virtualdj      | export -> `virtualdj/inbox/<id>/`         |
| jbl_bandbox (planned)    | loopback capture -> `jbl_bandbox/captures/<id>/` |

`<id>` = MUSDB track folder name. Manual/capture stems that aren't present yet
are listed in `runs/<run>/pending.json` instead of failing the run.

## Run
```bash
# from this directory, so `benchmark` and `separators` are importable
pip install -r benchmark/requirements.txt
export MUSDB18_HQ_ROOT=/path/to/musdb18hq        # dataset lives elsewhere
python -m benchmark.cli --profiles 4stem 2stem --window 1.0 --run-dir runs/first
```
Outputs: `runs/<run>/results.parquet`, `summary.csv`, per-track `metrics/*.json`,
`pending.json`.

## Notes that bite
- **Window**: 1.0 s = current SDX/SiSEC convention (museval default is 2.0 s).
  Fixed per run and recorded — keep it constant for comparability to literature.
- **Alignment**: capture/GUI latency beyond ~11.6 ms breaks SDR; the evaluator
  coarse-aligns via cross-correlation before museval.
- **Spleeter**: its own venv; pass `--spleeter-python /path/to/venv/python`.
