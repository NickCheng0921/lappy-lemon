#!/usr/bin/env bash
#
# AUX-in -> delay line -> speaker/headphone for the DA7212 Audio Board (A).
#
# Buffers incoming audio and streams it back out DELAY_MS milliseconds later.
# SoX's `delay` effect is a true streaming delay line: it pads the start of
# the stream with silence, so from then on everything runs that far behind.
#
# Architecture: arecord | sox | aplay -- three CONCURRENT processes with pipes
# as elastic buffers. Do NOT collapse this into a single sox full-duplex call:
# sox reads and writes serially, so capture overflows while it's writing and
# playback starves while it's reading -> endless "under-run"/"over-run" pairs.
#
# Usage (run ON the Pi):
#   ./delay.sh              # use DELAY_MS below
#   ./delay.sh 2500         # override: delay by 2500 ms
#
# Memory cost is trivial: 48kHz stereo S16 ~= 192 KB per second of delay.

set -euo pipefail

# ------------------------- delay knobs --------------------------------
DELAY_MS=${1:-1000}   # how far behind the output runs, in milliseconds
ECHO_MODE=0           # 0 = delayed only, 1 = mix live + delayed (slapback echo)
ECHO_DECAY=0.5        # if ECHO_MODE=1, level of the delayed copy (0-1)

# Gain stages, in dB. Use dB not % -- this codec's % scale is wildly
# non-linear (Headphone 40% = -32dB but 92% = +1dB), which makes the
# digital path sound far quieter than passthrough.sh's analog bypass.
#
# If it's too quiet, turn things up in THIS order. Each stage amplifies the
# noise of everything before it, so gain added late = hiss:
#   1. the PC's own output volume   (free, no added noise)
#   2. AUX_DB    (earliest stage, best signal-to-noise)
#   3. MIXIN_DB  (last resort -- boosts the front-end's own noise)
# OUT_DB is after the DAC, so it scales signal and hiss equally.
AUX_DB=0              # 'Aux' input PGA      (max +15dB)
MIXIN_DB=0            # 'Mixin PGA' into ADC (max +10.5dB; noisiest knob)
OUT_DB=0              # 'Headphone'/'Lineout' output (negative = quieter)

PERIOD_MS=20          # ALSA period size; smaller = lower latency
BUFFER_MS=80          # ALSA buffer size; must be a few periods. Raise both
                      # if you get under-run/over-run warnings.
# -----------------------------------------------------------------------

CARD=Zero            # aplay -l -> card: Zero [RPi Codec Zero] (da7213)
DEV=hw:CARD=$CARD,DEV=0
RATE=48000
FMT=S16_LE
STATE=$(mktemp /tmp/delay-state.XXXXXX)
CFG=~/da7212-config/All-input-output.state

restore() {
    trap - EXIT INT TERM        # don't run twice (Ctrl-C fires INT then EXIT)
    echo
    echo "Restoring previous mixer state..."
    sudo alsactl restore -f "$STATE" 2>/dev/null || true
    sudo rm -f "$STATE"         # alsactl store ran as root, so rm must too
    echo "Done."
}

sudo alsactl store -f "$STATE"
trap restore EXIT INT TERM

# Start from Waveshare's known-good FULL-DUPLEX state. Input-only snapshots
# (Aux-input.state) mute the DAC and unroute Mixout -> silence on playback.
sudo alsactl restore -f "$CFG" 2>/dev/null

# ...but "All-input" means ALL inputs: it sums both onboard mics at +30dB into
# the right Mixin channel. Amplifying that = loud static + room pickup.
# Killing the mic paths drops the noise floor ~15dB. AUX only:
for m in 'Mixin Left Mic 1'  'Mixin Left Mic 2' \
         'Mixin Right Mic 1' 'Mixin Right Mic 2' \
         'Onboard MIC' 'MIC Jack' 'DMIC'; do
    amixer -c "$CARD" sset "$m" off >/dev/null 2>&1 || true
done
amixer -c "$CARD" sset 'Mic 1' 0% off >/dev/null 2>&1 || true
amixer -c "$CARD" sset 'Mic 2' 0% off >/dev/null 2>&1 || true

# Input: AUX -> Mixin -> ADC.  Mixin PGA ships at 0%, so it must be raised.
amixer -c "$CARD" sset 'Aux' "${AUX_DB}dB" on          >/dev/null
amixer -c "$CARD" sset 'Mixin Left Aux Left' on        >/dev/null
amixer -c "$CARD" sset 'Mixin Right Aux Right' on      >/dev/null
amixer -c "$CARD" sset 'Mixin PGA' "${MIXIN_DB}dB" on  >/dev/null
# Output volume (routing already correct from the state file above).
amixer -c "$CARD" sset 'Headphone' "${OUT_DB}dB" on    >/dev/null
amixer -c "$CARD" sset 'Lineout' "${OUT_DB}dB" on      >/dev/null

if [[ "$ECHO_MODE" == "1" ]]; then
    # echos <gain-in> <gain-out> <delay-ms> <decay>: live signal plus one
    # delayed copy mixed on top.  (This effect DOES take milliseconds.)
    EFFECT=(echos 0.8 0.9 "$DELAY_MS" "$ECHO_DECAY")
    echo "Echo mode: live + copy delayed by ${DELAY_MS}ms (decay ${ECHO_DECAY})"
else
    # delay <per-channel offset>: the whole stream comes out this late.
    # sox positions are in SECONDS (no "ms" suffix), one per channel.
    DELAY_S=$(awk "BEGIN{printf \"%.3f\", $DELAY_MS/1000}")
    EFFECT=(delay "$DELAY_S" "$DELAY_S")
    echo "Delay mode: output runs ${DELAY_MS}ms behind the input."
fi

echo "Ctrl-C to stop and restore previous mixer state."

RAW=(-t raw -r $RATE -e signed -b 16 -c 2)
PT=$((PERIOD_MS * 1000))
BT=$((BUFFER_MS * 1000))

arecord -D "$DEV" -f $FMT -r $RATE -c 2 -t raw \
        --period-time=$PT --buffer-time=$BT 2>/dev/null \
  | sox -q "${RAW[@]}" - "${RAW[@]}" - "${EFFECT[@]}" \
  | aplay -D "$DEV" -f $FMT -r $RATE -c 2 -t raw \
        --period-time=$PT --buffer-time=$BT 2>/dev/null
