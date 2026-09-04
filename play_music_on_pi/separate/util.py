"""Shared logging. Everything writes progress to stderr so stdout
stays clean for gain dumps and piped audio."""

import sys


def log(*a):
    print(*a, file=sys.stderr, flush=True)
