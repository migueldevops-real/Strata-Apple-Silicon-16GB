#!/bin/bash
cd "$(dirname "$0")" || exit 1
if [ ! -d .venv ]; then echo "Run ./setup-macos.sh first."; exit 1; fi
exec .venv/bin/python tools/strata_menubar.py
