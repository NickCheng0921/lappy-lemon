# lappy-lemon

# Setting up the vendor directory to run demucs

Build w/ python 3.11, example uv command: `uv venv --python 3.11 .venv`

Populate libs with: `uv pip install -r vendor/requirements.txt`, `uv pip install "numpy<2"`
 - requirements doesn't give strict np version, will break when you go to run demucs without downgrade

# Running demucs

`python -m demucs -n htdemucs ../test_demucs/klk_op2.mp3 -o ../test_demucs/outs/ --mp3`

I can split a 2 minute track w/ htdemucs in 22 seconds, 7600x + 4090

# Misc Info

**Installing ffmpeg**

sudo apt update && sudo apt install -y ffmpeg

**WSL + Windows**

When working w/ WSL on windows committed files, run `git config core.autocrlf input` to prevent wsl from showing line endings as content change