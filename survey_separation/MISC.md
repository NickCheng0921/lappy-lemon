# Benchmark current audio separation tooling

### State of Stem Split Tools (2026)

Ultimate Vocal Remover - https://ultimatevocalremover.com/
-  selects from a set of models
- htdemucs v4 and htdemucs_ft v4 are the current best

JBL BandBox
- audio speaker containing real time step separation (identical to what I'm trying to do)

FL Studio
- mixing software w/ AI stem splitting

VirtualDJ
- dj software w/ AI stem splitting

Spleeter
- stem split github ran by Deezer, an audio streaming site (like spotify)

### Benchmark

Evaluate models on separation quality on MUSDB and investigate model architectures to compare to my distilled demucs model.

Goal is to see how far the gap is between my project + current products.