#!/bin/bash
# Lookup-logged replays with PR #63851 (patches/pr63851.diff) applied to the working tree.
# Do not edit vllm/ while this runs: the second server start would pick up the edits.
S=$(dirname "$(readlink -f "$0")")
TAG=n8_pr63851 "$S/cc_run.sh" --sessions 8 --max-req 40
TAG=n4_pr63851 "$S/cc_run.sh" --sessions 4 --max-req 40
echo ALL_DONE
