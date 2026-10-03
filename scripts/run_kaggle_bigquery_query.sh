#!/usr/bin/env bash
# Reproduces the real-data pull used to find/fix the citation-regex bug.
#
# The real USPTO Office Action Research Dataset (patents-public-data.
# uspto_oce_office_actions on Google BigQuery, mirrored on Kaggle as
# bigquery/uspto-oce-office-actions) is NOT downloadable as flat files via
# the Kaggle Datasets API -- `kaggle datasets download` 404s on it, because
# it's a BigQuery-only integration. Kaggle *does* give any pushed kernel
# free, pre-authenticated BigQuery access to it, so this script runs the
# query headlessly as a Kaggle kernel (via `kaggle kernels push`) and pulls
# the resulting CSVs back down -- no separate GCP project/credentials needed,
# just a Kaggle API token. See docs/DATA_SOURCES.md for the full story.
#
# Prerequisites:
#   pip install kaggle
#   Kaggle API token at ~/.kaggle/access_token (kaggle.com/settings -> API)
#
# Usage:
#   ./scripts/run_kaggle_bigquery_query.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KERNEL_DIR="$ROOT/scripts/kaggle_bigquery_kernel"
OUT_DIR="$ROOT/data/raw/kaggle_uspto_oce"
KERNEL_ID=$(python3 -c "import json; print(json.load(open('$KERNEL_DIR/kernel-metadata.json'))['id'])")

mkdir -p "$OUT_DIR"

echo "Pushing kernel $KERNEL_ID ..."
kaggle kernels push -p "$KERNEL_DIR"

echo "Waiting for kernel to finish ..."
for _ in $(seq 1 30); do
    status=$(kaggle kernels status "$KERNEL_ID" 2>&1)
    echo "  $status"
    if echo "$status" | grep -qE "COMPLETE|ERROR"; then
        break
    fi
    sleep 10
done

echo "Downloading output to $OUT_DIR ..."
kaggle kernels output "$KERNEL_ID" -p "$OUT_DIR"

echo "Done. Files: $(ls "$OUT_DIR")"
