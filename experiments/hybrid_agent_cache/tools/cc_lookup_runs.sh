#!/bin/bash
# Per-group hit decomposition runs (A vs B): 8 sessions, then 4 sessions as control.
# The probe writes per-lookup records to results/cctrace/probe_cc_<tag>.jsonl.lookups.
S=$(dirname "$(readlink -f "$0")")
TAG=n8_lookup "$S/cc_run.sh" --sessions 8 --max-req 40
TAG=n4_lookup "$S/cc_run.sh" --sessions 4 --max-req 40
echo ALL_DONE
