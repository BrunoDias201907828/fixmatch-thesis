#!/bin/bash

METHOD="FixMatch_new_multiple"
CONFIDENCE_THRESHOLDS="0.75 0.80 0.85 0.90 0.95"
TYPES="cosine euclidean"

for CONFIDENCE_THRESHOLD in $CONFIDENCE_THRESHOLDS; do
    for TYPE in $TYPES; do
        OUTPUT_FILE="${METHOD}-${CONFIDENCE_THRESHOLD}-${TYPE}.txt"
        echo "Running $METHOD with confidence_threshold=$CONFIDENCE_THRESHOLD and type=$TYPE"
        python train.py model-$METHOD.pth $METHOD --confidence_threshold $CONFIDENCE_THRESHOLD --type $TYPE > $OUTPUT_FILE
    done
done
