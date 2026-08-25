# Passthrough Debug Log

Gotchas hit while building the AUX-in effect scripts on the DA7212 board.

---

## Silence + endless `under-run` / `over-run` from sox

**Symptom:** `./delay.sh` ran with no audio out, spewing alternating
`sox WARN alsa: under-run` / `sox WARN alsa: over-run`. Two unrelated bugs.

### Bug 1 — input-only `.state` files mute the whole output chain

Applying `Aux-input.state` to set up capture silently wrecked playback:

| control                  | left at        |
|--------------------------|----------------|
| `DAC`                    | −99999 dB (muted) |
| `Mixout Left/Right DAC`  | off (DAC unrouted from output mixer) |
| `HP Jack`                | off            |
| `Lineout`                | 0%             |
| `Mixin PGA`              | 0% (capture only −34 dB peak) |

Setting `Headphone`/`Lineout` to "on" isn't enough — everything *upstream* of
them was dead. **Fix:** always base on `All-input-output.state` (the known-good
full-duplex snapshot), then adjust volumes on top. Never use an input-only
snapshot when you also need playback. `Mixin PGA` ships at 0% and must be
raised explicitly or capture is near-silent.

Useful one-liner to inspect the chain:
```bash
for c in 'DAC' 'Mixout Left DAC Left' 'Headphone' 'Lineout' 'HP Jack' 'Mixin PGA'; do
  printf '%-24s ' "$c"; amixer -c Zero sget "$c" | tail -1; done
```

### Bug 2 — one sox process can't do full duplex

`sox -t alsa <dev> -t alsa <dev>` reads and writes **serially in one loop**:
while it's writing, the capture buffer overflows (over-run); while it's
reading, the playback buffer starves (under-run). Hence the alternating pairs
— no buffer size fixes it, it's the architecture.

**Fix:** split into concurrent processes with pipes as elastic buffers:
```bash
arecord -D "$DEV" -f S16_LE -r 48000 -c 2 -t raw | sox -q -t raw ... - -t raw ... - <effect> | aplay -D "$DEV" ...
```

### Also
- sox `delay` positions are in **seconds** (`1.000`), not `1000ms`. The
  `echos` effect *does* take milliseconds — easy to mix up.
- `alsactl store` writes as root, so the temp state file needs `sudo rm`.
- Traps: `Ctrl-C` fires INT *then* EXIT — `trap - EXIT INT TERM` inside the
  handler stops it running twice.
