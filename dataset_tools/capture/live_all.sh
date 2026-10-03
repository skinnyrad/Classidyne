#!/bin/zsh
# Receive-only live capture suite (HackRF RX first, then RTL-SDR off-air). Resumable per target.
cd "${0:A:h}"
PY=../../venv-classidyne/bin/python  # repo venv
run() { $PY run_live.py "$1" --frames "$2" --per-center 2 --min-score 45 --seed $RANDOM; }
caffeinate -u -t 2
run bluetooth-esp32 60
caffeinate -u -t 2; run cellular 50
caffeinate -u -t 2; run atsc-ota 60
caffeinate -u -t 2; run hdmi-leak 40
caffeinate -u -t 2; run ads-b-ota 30
caffeinate -u -t 2; run fm-ota 30
caffeinate -u -t 2; run airband-ota 20
caffeinate -u -t 2; run ais-ota 20
caffeinate -u -t 2; run pocsag-ota 20
caffeinate -u -t 2; run vor-ota 15
echo "live suite done"
