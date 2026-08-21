#!/usr/bin/env bash
#
# SSH to the Raspberry Pi and run commands.
#
# Usage:
#   ./pi.sh                       # open an interactive shell on the pi
#   ./pi.sh "aplay song.wav"      # run a single command on the pi
#   ./pi.sh -f local_commands.sh  # run a local script file on the pi
#
# Configure the connection with env vars or by editing the defaults below:
#   PI_USER   ssh username on the pi   (default: pi)
#   PI_HOST   hostname or IP of the pi (default: raspberrypi.local)
#   PI_PORT   ssh port                 (default: 22)

set -euo pipefail

PI_USER="${PI_USER:-nicknack}"
PI_HOST="${PI_HOST:-nickpi.local}"
PI_PORT="${PI_PORT:-22}"

SSH_TARGET="${PI_USER}@${PI_HOST}"
SSH_OPTS=(-p "${PI_PORT}")

if [[ "${1:-}" == "-f" ]]; then
    # Pipe a local script file to the pi's shell.
    script_file="${2:?usage: ./pi.sh -f <script-file>}"
    ssh "${SSH_OPTS[@]}" "${SSH_TARGET}" 'bash -s' < "${script_file}"
elif [[ $# -gt 0 ]]; then
    # Run whatever was passed as a single remote command.
    ssh "${SSH_OPTS[@]}" "${SSH_TARGET}" "$@"
else
    # No args: drop into an interactive shell on the pi.
    ssh "${SSH_OPTS[@]}" "${SSH_TARGET}"
fi
