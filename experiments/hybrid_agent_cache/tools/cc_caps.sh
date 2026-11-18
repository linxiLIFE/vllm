#!/bin/bash
# Replay 8 sessions with --max-num-active-seqs caps; detached from the tool timeout.
S=$(dirname "$(readlink -f "$0")")
for c in 4 6; do
  TAG=n8_cap$c EXTRA="--max-num-active-seqs $c" "$S/cc_run.sh" --sessions 8 --max-req 40
done
echo ALL_DONE
