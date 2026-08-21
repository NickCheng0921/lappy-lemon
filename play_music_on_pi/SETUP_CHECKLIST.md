# DA7212 Audio Board (A) — Pi 5 Setup Checklist

Board: Waveshare **DA7212 Audio Board (A)** (docs.waveshare.com/DA7212-Audio-Board-A).
It's a HAT+ with an ID EEPROM — the Pi **auto-detects it at boot** and loads the
in-kernel `snd-soc-da7213` driver. **No driver install, no dtoverlay line needed.**
It shows up as card `RPi Codec Zero` / `da7213-hifi`.

Tested on: Pi 5, Raspberry Pi OS Bookworm 64-bit, kernel 6.12.
Attach the HAT **before powering on** or it won't be detected.

## On the laptop (reconnect to a fresh image)

```bash
ssh-keygen -R nickpi.local          # clear old host key
ssh-copy-id nicknack@nickpi.local   # skip if key pasted in Imager
```

- `./pi.sh "<cmd>"` in this folder runs a command on the Pi
  (defaults: `nicknack@nickpi.local`; override with `PI_HOST=<ip>`).
- If `.local` won't resolve in Git Bash: `ping nickpi.local` from Windows, use the IP.

## On the Pi

```bash
# 1. Verify auto-detection (HAT EEPROM + sound card)
tr -d '\0' < /proc/device-tree/hat/product   # → "DA7212 Audio HAT"
aplay -l                                     # → card N: Zero [RPi Codec Zero] ... da7213-hifi

# 2. Make sure NOTHING ELSE claims I2S or i2c addr 0x1a — this blocks the HAT.
#    config.txt must NOT have: dtoverlay=wm8960-soundcard / i2s-mmap / dtparam=i2s=on
grep -nE 'wm8960|i2s' /boot/firmware/config.txt          # want: no uncommented hits
systemctl list-unit-files | grep -iE 'wm8960|seeed'      # want: nothing enabled
# If a seeed/wm8960 service exists (leftover from an old HAT), kill it:
#   sudo systemctl disable --now seeed-voicecard.service && sudo reboot
# Symptoms of the conflict: dmesg "gpio18 already requested", i2c "-16" at 0x1a,
# and playback dying with Input/output error (-5) / "dma2chan2 non-idle".

# 3. Configure the mixer with Waveshare's official ALSA state files
wget 'https://gitee.com/waveshare/DA7212-Audio-Board-A/raw/master/examples/DA7212-Audio-Board-A-Config.zip'
unzip DA7212-Audio-Board-A-Config.zip -d da7212-config
cd da7212-config && sudo alsactl restore -f ./All-input-output.state
# (other .state files: Headphone-input-output, Single/Dual-speaker-output, MIC inputs)

# 4. Set a sane volume and PERSIST it (state files default to ~90% = loud)
amixer -c 0 sset Headphone 40%    # 3.5mm jack
amixer -c 0 sset Lineout 40%      # speaker terminals
sudo alsactl store                # restored automatically every boot

# 5. Test tone
speaker-test -D plughw:0 -c 2 -t sine -f 440 -l 1

# 6. Play a song
sudo apt install -y mpg123        # if missing
mpg123 -a plughw:0 ~/Music/yrshka_sunny.mp3
```

## Notes
- Card number can shuffle; when in doubt use the name from `aplay -l`
  (`plughw:CARD=Zero`) instead of `plughw:0`.
- mpg123's `+`/`-` keys are per-process software gain — never saved. Use the
  ALSA mixer (`amixer` / `alsamixer` + `sudo alsactl store`) for persistent
  volume, or start quieter with `mpg123 -f 13000 ...` (full scale = 32768).
- Codec is DA7212 at i2c-1 `0x1a`; HAT EEPROM at i2c-0 `0x50`. `sudo i2cdetect -y 1`
  showing `UU` at 1a = driver bound (good).
- The old WM8960 HAT notes live in `past_try.txt` — that board was likely
  defective; none of its driver/overlay steps apply to this one.
