#!/usr/bin/env bash
#
# Rebuild ~/demucs_opt on a freshly flashed Pi, from the laptop.
#
# Run from the repo root:
#   bash play_music_on_pi/setup_pi.sh
#   PI_HOST=<ip> PI_USER=<user> bash play_music_on_pi/setup_pi.sh
#
#   PI_HOST   hostname or IP (default: nickpi.local)
#   PI_USER   ssh user, also the boot service instance (default: nicknack)
#
# Image first (Raspberry Pi Imager): Raspberry Pi OS Bookworm 64-bit, hostname
# and user matching the above, SSH on with your public key, Wi-Fi set. HAT on
# before first boot. Then: ssh-keygen -R <host>, and ssh-copy-id if no key in Imager.
#
# Idempotent: safe to re-run after a failed step. Makes no sound.

set -euo pipefail

cd "$(dirname "$0")/.."
export PI_HOST=${PI_HOST:-nickpi.local}
export PI_USER=${PI_USER:-nicknack}
PI="bash play_music_on_pi/pi.sh"
MODEL=pi_optimize_demucs/models/student_final_slim.onnx
STUBS=survey_separation/separators/htdemucs_distill/stubs

[[ -f $MODEL ]] || { echo "missing $MODEL (gitignored; restore from backup)"; exit 1; }

# mDNS drops mid-run (the model transfer especially); resolve once, pin the IP.
if [[ $PI_HOST == *.local ]]; then
    ip=$($PI "hostname -I" | awk '{print $1}')
    [[ -n $ip ]] || { echo "can't reach $PI_HOST; pass PI_HOST=<ip>"; exit 1; }
    PI_HOST=$ip
fi
echo "== using $PI_HOST"

echo "== system packages, i2c"
$PI "sudo apt-get update -qq && sudo apt-get install -y -qq python3-venv ffmpeg unzip alsa-utils i2c-tools sox \
  && sudo raspi-config nonint do_i2c 0 && sudo usermod -aG i2c,audio \$USER"

echo "== code"
tar czf /tmp/lappy_src.tgz --exclude=__pycache__ vendor/demucs profile/onnx_export.py \
    -C play_music_on_pi separate passthrough/passthrough.sh passthrough/delay.sh passthrough/chorus.sh \
    -C ../$(dirname $STUBS) stubs
$PI "mkdir -p ~/demucs_opt && cd ~/demucs_opt && tar xzf - && sed -i 's/\r$//' passthrough/*.sh && chmod +x passthrough/*.sh" < /tmp/lappy_src.tgz

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

echo "== DA7212 mixer (Waveshare state, mics off, Aux 0dB, 40% out)"
$PI "cd ~ && [ -d da7212-config ] || { wget -q 'https://gitee.com/waveshare/DA7212-Audio-Board-A/raw/master/examples/DA7212-Audio-Board-A-Config.zip' \
  && unzip -q DA7212-Audio-Board-A-Config.zip -d da7212-config; } \
  ; sudo alsactl restore -f da7212-config/All-input-output.state; \
  for m in 'Mixin Left Mic 1' 'Mixin Left Mic 2' 'Mixin Right Mic 1' 'Mixin Right Mic 2' 'Onboard MIC' 'MIC Jack'; do \
    amixer -q -c Zero sset \"\$m\" off; done; \
  amixer -q -c Zero sset 'Mic 1' 0% off; amixer -q -c Zero sset 'Mic 2' 0% off; amixer -q -c Zero sset Aux 0dB; \
  amixer -q -c Zero sset Headphone 40% && amixer -q -c Zero sset Lineout 40% && sudo alsactl store" \
  || echo "!! mixer step failed -- is the HAT detected? (aplay -l)"

echo "== boot service (enabled, not started)"
$PI "sudo cp ~/demucs_opt/separate/lappy-separate@.service /etc/systemd/system/ \
  && sudo systemctl daemon-reload && sudo systemctl enable lappy-separate@\$USER"

echo "== checks"
$PI "tr -d '\0' < /proc/device-tree/hat/product; echo; aplay -l | grep -i zero || echo '!! no Zero card'; \
  grep -nE '^[^#]*(wm8960|i2s)' /boot/firmware/config.txt && echo '!! stale wm8960/i2s lines in config.txt'; \
  ls -la ~/demucs_opt ~/demucs_opt/separate"

echo "== selftest (structural, no audio device)"
$PI "cd ~/demucs_opt/separate && PYTHONPATH=../stubs ../.venv/bin/python selftest.py"

echo
echo "done. reboot once for i2c + group membership:  $PI 'sudo reboot'"
