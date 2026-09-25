# separators/

One subpackage per tool. Each exposes `build() -> Separator` and is registered in
`separators/__init__.py:default_separators()`.

Add a tool:
1. `mkdir separators/<tool>` with `__init__.py` + `adapter.py`.
2. In `adapter.py`, either:
   - subclass nothing and wrap a CLI (see `spleeter/adapter.py`), or
   - reuse `_demucs_base.DemucsSeparator` (CLI models), or
   - reuse `_file_bases.InboxSeparator` (GUI export) / `CaptureSeparator` (recorded).
3. Declare `produces` with canonical names: vocals/drums/bass/other, and/or
   `accompaniment` for 2-stem tools.
4. Add its `build()` to `default_separators()`.

`inbox/` (manual) and `captures/` (recorded) folders hold `<track_id>/<stem>.wav`.
