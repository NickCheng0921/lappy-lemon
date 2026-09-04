"""The dB ladder: how a fader position becomes a linear gain.

Loudness tracks dB, so travel is mapped in decibels; a fader linear in
amplitude is 14x more sensitive at one end than the other."""

import math


# ----------------------------------------------------------------- mix knobs
GAINS = {"drums": 1.0, "bass": 1.0, "other": 1.0, "vocals": 1.0}

# The travel is mapped in dB, not in amplitude. Loudness tracks dB, so a fader
# that is linear in amplitude is wildly uneven: the same physical distance is
# 6 dB near silence and 0.4 dB near the ceiling, i.e. 14x more effect at one
# end of the slider than the other. STEP_DB survives the move from keys to
# pots because MUTE_FLOOR_DB is still derived from it.
STEP_DB = 1.5
GAIN_MAX_DB = 12.0  # boost ceiling; the 4 stems are summed and hard-clipped
# at +/-1.0 in to_int16(), so more than this is mush

# A dB fader never reaches zero on its own, so silence has to be a decision
# rather than a limit. The keyboard build counted presses (ten down from unity
# kills a stem, the ninth is the quietest audible setting); a pot has no press
# count, so POT_MUTE_POS declares the bottom of the travel off and the usable
# scale ends one notch above it, here, at MUTE_FLOOR_DB.
PRESSES_TO_MUTE = 10
MUTE_FLOOR_DB = -STEP_DB * (PRESSES_TO_MUTE - 1)
GAIN_MIN, GAIN_MAX = 0.0, 10.0 ** (GAIN_MAX_DB / 20.0)
# ----------------------------------------------------------------------------

SOURCES = ["drums", "bass", "other", "vocals"]

# Five slide pots on two ADS1115s (4 channels each, addresses set by the ADDR
# pin: GND=0x48, VDD=0x49). Unused ADC channels must be tied to GND -- a
# floating input reads as a convincing mid-scale position, not as zero.
POT_MAP_DEFAULT = "0x48:0,1,2,3 0x49:0"
POT_UNITY_POS = 0.80  # travel fraction that means unity gain
POT_MUTE_POS = 0.02  # below this a fader is off, not merely quiet
POT_SMOOTH = 0.5  # one-pole EMA on position; 1.0 disables smoothing

# Travel is quantised into detents, so ADC noise cannot move the gain at all.
# Hysteresis is what makes that work -- without it a fader parked on a boundary
# flips between two detents, which is worse than the jitter it replaces.
POT_STEPS = 20  # 5% of travel each
POT_HYST = 0.25  # must overshoot the midpoint by this much to move

def detent(pos, held, steps=POT_STEPS, hyst=POT_HYST):
    """Snap position 0..1 to one of `steps` detents. `held` is the latched
    index (None first time); noise below 0.5+hyst steps cannot dislodge it."""
    x = min(max(pos, 0.0), 1.0) * steps
    if held is None:
        return int(round(x))
    if abs(x - held) > 0.5 + hyst:
        return int(min(max(round(x), 0), steps))
    return held

def pot_gain(pos):
    """Fader position (0..1) -> linear gain, on the range the keys used.

    The top of the travel above POT_UNITY_POS is boost (0 .. GAIN_MAX_DB) and
    everything below it is cut (MUTE_FLOOR_DB .. 0). Both halves are linear in
    dB rather than in amplitude, for the reason STEP_DB exists: a fixed
    amplitude move is 6 dB near silence and 0.4 dB near the ceiling, so a
    linear-in-amplitude fader would do 14x more at one end than the other.

    Silence stays a decision rather than a limit, as it was for the keys.
    A dB fader never reaches zero, so the bottom POT_MUTE_POS of the travel is
    declared off; the usable scale ends one notch above it at MUTE_FLOOR_DB.
    """
    if pos <= POT_MUTE_POS:
        return GAIN_MIN
    if pos >= POT_UNITY_POS:
        frac = (pos - POT_UNITY_POS) / max(1.0 - POT_UNITY_POS, 1e-9)
        db = min(1.0, frac) * GAIN_MAX_DB
    else:
        frac = (pos - POT_MUTE_POS) / (POT_UNITY_POS - POT_MUTE_POS)
        db = MUTE_FLOOR_DB * (1.0 - frac)
    return min(GAIN_MAX, 10.0 ** (db / 20.0))
