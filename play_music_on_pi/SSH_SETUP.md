# Passwordless SSH to the Pi (key-based auth)

The idea: you make a **key pair** — a private key that stays on your laptop, and a
public key you copy to the Pi. After that, SSH logs in with the key and never asks
for a password. You never share the private key (not with me, not with anyone).

## 1. Generate a key pair (once, on your laptop)

In Git Bash:

```bash
ssh-keygen -t ed25519 -C "lappy-lemon-pi"
```

- Press Enter to accept the default location (`~/.ssh/id_ed25519`, i.e. `C:\Users\nicks\.ssh\id_ed25519`).
- You can set a passphrase (recommended) or leave it empty for fully hands-off login.

This creates two files:
- `id_ed25519`      → **private key, keep secret, never copy off the laptop**
- `id_ed25519.pub`  → public key, safe to share/copy to the Pi

## 2. Copy the public key to the Pi

Git Bash ships `ssh-copy-id`, so the easy way (you'll enter the Pi password this
ONE last time):

```bash
ssh-copy-id nicknack@nickpi.local

ssh-keygen -R nickpi.local; ssh-keygen -R 10.0.0.101

# above may fail as DNS won't resolve, ping nickpi.local and manually add IP
ssh-copy-id nicknack@10.0.0.101
```

If `ssh-copy-id` isn't available, do it manually instead:

```bash
cat ~/.ssh/id_ed25519.pub | ssh nicknack@nickpi.local "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
```

Replace `nicknack@nickpi.local` with your Pi's user/host if different.

## 3. Test it

```bash
ssh nicknack@nickpi.local
```

It should log in without asking for a password (only the key passphrase, if you set one).

## 4. Use the helper script

```bash
PI_HOST=raspberrypi.local PI_USER=pi ./pi.sh "aplay /home/pi/song.wav"
```

## Notes on keeping me (Claude) out of your secrets

- I only ever run the `pi.sh` script, which uses your locally-stored key.
- The private key never appears in any command, script, or chat — SSH reads it
  directly from `~/.ssh/`. Nothing here needs your password.
