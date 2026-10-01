#!/usr/bin/env bash
#
# Rebuild ~/demucs_opt on a freshly flashed Pi, from the laptop.
#
# Run from the repo root:
#   PI_HOST=<ip> bash play_music_on_pi/setup_pi.sh
#
# Image first (Raspberry Pi Imager): Raspberry Pi OS Bookworm 64-bit, hostname
# nickpi, user nicknack, SSH on with your public key, Wi-Fi set. HAT on before
# first boot. Then: ssh-keygen -R <host>, and ssh-copy-id if no key in Imager.
#
# Idempotent: safe to re-run after a failed step. Makes no sound.

set -euo pipefail

cd "$(dirname "$0")/.."
PI="bash play_music_on_pi/pi.sh"
MODEL=pi_optimize_demucs/models/student_final_slim.onnx
STUBS=survey_separation/separators/htdemucs_distill/stubs

[[ -f $MODEL ]] || { echo "missing $MODEL (gitignored; restore from backup)"; exit 1; }

# mDNS drops mid-run (the model transfer especially); resolve once, pin the IP.
if [[ ${PI_HOST:-nickpi.local} == *.local ]]; then
    ip=$($PI "hostname -I" | awk '{print $1}')
    [[ -n $ip ]] || { echo "can't reach ${PI_HOST:-nickpi.local}; pass PI_HOST=<ip>"; exit 1; }
    export PI_HOST=$ip
fi
echo "== using $PI_HOST"

echo "== system packages, i2c"
$PI "sudo apt-get update -qq && sudo apt-get install -y -qq python3-venv ffmpeg unzip alsa-utils i2c-tools \
  && sudo raspi-config nonint do_i2c 0 && sudo usermod -aG i2c,audio \$USER"

echo "== code"
tar czf /tmp/lappy_src.tgz --exclude=__pycache__ vendor/demucs profile/onnx_export.py \
    -C play_music_on_pi separate \
    -C ../$(dirname $STUBS) stubs
$PI "mkdir -p ~/demucs_opt && cd ~/demucs_opt && tar xzf -" < /tmp/lappy_src.tgz

echo "== model"
$PI "cat > ~/demucs_opt/student_final_slim.onnx" < $MODEL
want=$(stat -c %s $MODEL)
got=$($PI "stat -c %s ~/demucs_opt/student_final_slim.onnx")
[[ $want == "$got" ]] || { echo "model truncated: $got / $want bytes, re-run"; exit 1; }

echo "== venv"
$PI "cd ~/demucs_opt && { [ -d .venv ] || python3 -m venv .venv; } \
  && .venv/bin/pip install -q --upgrade pip \
  && .venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu \
  && .venv/bin/pip install -q onnxruntime numpy einops julius omegaconf dora-search tqdm soundfile"

echo "== DA7212 mixer (Waveshare state, then 40% out)"
$PI "cd ~ && [ -d da7212-config ] || { wget -q 'https://gitee.com/waveshare/DA7212-Audio-Board-A/raw/master/examples/DA7212-Audio-Board-A-Config.zip' \
  && unzip -q DA7212-Audio-Board-A-Config.zip -d da7212-config; } \
  ; sudo alsactl restore -f da7212-config/All-input-output.state; \
  amixer -q -c Zero sset Headphone 40% && amixer -q -c Zero sset Lineout 40% && sudo alsactl store" \
  || echo "!! mixer step failed -- is the HAT detected? (aplay -l)"

echo "== checks"
$PI "tr -d '\0' < /proc/device-tree/hat/product; echo; aplay -l | grep -i zero || echo '!! no Zero card'; \
  grep -nE '^[^#]*(wm8960|i2s)' /boot/firmware/config.txt && echo '!! stale wm8960/i2s lines in config.txt'; \
  ls -la ~/demucs_opt ~/demucs_opt/separate"

echo "== selftest (structural, no audio device)"
$PI "cd ~/demucs_opt/separate && PYTHONPATH=../stubs ../.venv/bin/python selftest.py"

echo
echo "done. reboot once for i2c + group membership:  $PI 'sudo reboot'"
