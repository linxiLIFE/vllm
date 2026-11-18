#!/bin/bash
# Same as cc_lookup_runs.sh but with PR #61282 (rebased, patches/pr61282_src_only.diff)
# applied to the working tree. Compare against n8_lookup / n4_lookup.
S=$(dirname "$(readlink -f "$0")")
TAG=n8_pr61282 "$S/cc_run.sh" --sessions 8 --max-req 40
TAG=n4_pr61282 "$S/cc_run.sh" --sessions 4 --max-req 40
echo ALL_DONE
