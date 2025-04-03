#!/bin/bash

METHOD="FixMatch"
CONFIDENCE_THRESHOLDS="0.85 0.9 0.95"

for CONFIDENCE_THRESHOLD in $CONFIDENCE_THRESHOLDS; do
    OUTPUT_FILE="${METHOD}-${CONFIDENCE_THRESHOLD}.txt"
    echo "Running $METHOD with confidence_threshold=$CONFIDENCE_THRESHOLD"
    python train.py model-$METHOD.pth $METHOD --confidence_threshold $CONFIDENCE_THRESHOLD > $OUTPUT_FILE
done