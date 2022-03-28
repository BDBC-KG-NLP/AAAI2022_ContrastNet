#!/usr/bin/env bash

sbatch \
    -n 1 \
    -c 1 \
    --mem=100G \
    -J "paraphrase-evaluation" \
    -o "paraphrase-evaluation.log" \
    ./utils/scripts/paraphrase/evaluate-paraphrase-diversity.sh