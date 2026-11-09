#!/usr/bin/env bash
# Fetch the three public prompt sources into data/raw/ (about 43 MB, mostly Alpaca).
set -euo pipefail
cd "$(dirname "$0")" && mkdir -p raw && cd raw
curl -sSL -o MultiJail.csv       https://huggingface.co/datasets/DAMO-NLP-SG/MultiJail/resolve/main/MultiJail.csv               # MIT
curl -sSL -o xstest_prompts.csv  https://huggingface.co/datasets/Paul/XSTest/resolve/main/xstest_prompts.csv                   # CC-BY-4.0
curl -sSL -o alpaca_cleaned.json https://huggingface.co/datasets/yahma/alpaca-cleaned/resolve/main/alpaca_data_cleaned.json   # CC-BY-4.0
ls -la
