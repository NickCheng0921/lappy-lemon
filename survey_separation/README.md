# survey_separation

Benchmark harness for music stem-separation tools on MUSDB18-HQ, scored with
BSS Eval v4 (museval). Datasets live **outside** this tree.

```
benchmark/     tool-agnostic harness (IO, alignment, museval, aggregation, CLI)
separators/    one subpackage per tool; each exposes build() -> Separator
```

## Design in one line
Adapters only *produce* stems (or ingest human-rendered / captured ones);
everything downstream sees a uniform `StemSet` and never knows which tool made it.

## Ingestion, by tool
| Tool           | How stems arrive                          | Profiles      |
|----------------|-------------------------------------------|---------------|
| htdemucs       | `demucs` CLI (auto)                       | 4stem, 2stem  |
| htdemucs_ft    | `demucs` CLI (auto)                       | 4stem, 2stem  |
| spleeter       | `spleeter` CLI, separate venv (auto)      | 4stem, 2stem  |
| flstudio       | GUI export -> `flstudio/inbox/<id>/`      | 4stem, 2stem  |
| virtualdj      | export -> `virtualdj/inbox/<id>/`         | 2stem         |
| jbl_bandbox (planned)    | loopback capture -> `jbl_bandbox/captures/<id>/` | 2stem  |

`<id>` = MUSDB track folder name. Manual/capture stems that aren't present yet
are listed in `runs/<run>/pending.json` instead of failing the run.

## Profiles
- **4stem**: vocals/drums/bass/other (Demucs, Spleeter, FL Studio).
- **2stem**: vocals vs accompaniment — the only apples-to-apples axis across
  *every* tool. Reference accompaniment is summed from drums+bass+other; a
  4-stem tool's accompaniment is summed from its own three non-vocal stems.

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
