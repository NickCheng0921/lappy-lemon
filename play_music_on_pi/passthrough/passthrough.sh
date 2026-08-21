#!/usr/bin/env bash
#
# AUX-in -> speaker/headphone passthrough for the DA7212 Audio Board (A).
#
# Default mode is ANALOG BYPASS: the DA7212 routes the Aux input straight
# into the output mixers inside the codec chip itself. No CPU, no buffers,
# effectively zero latency — a real "direct wire".
#
# Usage (run ON the Pi, or via ../pi.sh -f passthrough.sh):
#   ./passthrough.sh            # analog bypass until Ctrl-C, then restore
#   ./passthrough.sh loop       # software fallback: alsaloop capture->playback
#
# The previous mixer state is saved and restored on exit.

set -euo pipefail

CARD=Zero            # aplay -l -> card: Zero [RPi Codec Zero] (da7213)
STATE=$(mktemp /tmp/passthrough-state.XXXXXX)

restore() {
    echo
    echo "Restoring previous mixer state..."
    sudo alsactl restore -f "$STATE" 2>/dev/null || true
    rm -f "$STATE"
    echo "Done."
}

# Save current mixer state, restore it however we exit.
sudo alsactl store -f "$STATE"
trap restore EXIT INT TERM

if [[ "${1:-}" == "loop" ]]; then
    # ---- Software fallback: digitize Aux, play it back out. ----
    # Adds ADC->DAC + buffer latency (~tens of ms), but works if the
    # analog path misbehaves. Needs Aux routed into the ADC first.
    amixer -c "$CARD" sset 'Aux' 70% on            >/dev/null
    amixer -c "$CARD" sset 'Mixin Left Aux Left' on   >/dev/null
    amixer -c "$CARD" sset 'Mixin Right Aux Right' on >/dev/null
    echo "Software loop running (Ctrl-C to stop)..."
    alsaloop -C hw:CARD=$CARD -P hw:CARD=$CARD -t 10000 --sync=none
else
    # ---- Analog bypass: Aux -> output mixers, inside the codec. ----
    amixer -c "$CARD" sset 'Aux' 70% on               >/dev/null
    amixer -c "$CARD" sset 'Mixout Left Aux Left' on  >/dev/null
    amixer -c "$CARD" sset 'Mixout Right Aux Right' on >/dev/null
    # Make sure the outputs themselves are alive.
    amixer -c "$CARD" sset 'Headphone' on             >/dev/null
    amixer -c "$CARD" sset 'Lineout' on               >/dev/null
    echo "Analog bypass active: AUX in -> headphone/speaker."
    echo "Volume knobs: 'Aux' (input gain), 'Headphone'/'Lineout' (output)."
    echo "Ctrl-C to stop and restore previous mixer state."
    # Idle until interrupted; the audio path lives in the codec hardware.
    sleep infinity
fi
