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

---

## Way too quiet, then way too much hiss

**Symptom:** software path far quieter than the analog bypass; turning it up
buried everything in static. Two causes — a bad volume unit and open mics.

### The `%` scale is not linear — always set dB

`Headphone` at **40% = −32 dB**, but **92% = +1 dB**. A `OUT_VOL=40` that
looked like "40% volume" was throwing away ~33 dB (~45× in amplitude), while
passthrough.sh inherited ~92%. The extra ADC/DAC stages were a red herring —
they *add* gain. Use `amixer sset 'Headphone' 0dB`, never a percentage.

### `All-input-output.state` really does mean all inputs

It enables **both onboard mics at +30 dB into the right Mixin channel**. Every
dB of make-up gain was amplifying two open mics along with the music. Muting
the mic paths dropped the right channel from **−41.6 to −57.2 dB RMS (~15 dB)**:

```bash
for m in 'Mixin Left Mic 1' 'Mixin Left Mic 2' 'Mixin Right Mic 1' \
         'Mixin Right Mic 2' 'Onboard MIC' 'MIC Jack' 'DMIC'; do
  amixer -c Zero sset "$m" off; done
amixer -c Zero sset 'Mic 1' 0% off; amixer -c Zero sset 'Mic 2' 0% off
```

Matters double for chorus — an open mic near the speakers feeds back.

### Gain staging: add level EARLY

Each stage amplifies the noise of everything before it. To get louder, in
order: **PC volume** (free) → **`Aux` PGA** (first stage, best SNR) →
**`Mixin PGA`** last, it's the noisiest knob. Post-DAC output volume scales
signal and hiss equally, so it never improves SNR.
