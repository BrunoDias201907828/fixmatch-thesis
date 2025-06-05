#!/bin/bash

METHOD="FixMatch_DeepBilevel"
FREQUENCY_THRESHOLDS="0.75 0.80 0.85 0.90 0.95"
TYPES="cosine euclidean"
LABELS="250 4000 10000 20000"

for FREQUENCY_THRESHOLD in $FREQUENCY_THRESHOLDS; do
    for TYPE in $TYPES; do
        for LABEL in $LABELS; do
            OUTPUT_FILE="${METHOD}-${FREQUENCY_THRESHOLD}-${TYPE}-${LABEL}.txt"
            echo "Running $METHOD with frequency_threshold=$FREQUENCY_THRESHOLD, type=$TYPE and num-labeled=$LABEL"
            python3 train_gradients.py model-$METHOD-$FREQUENCY_THRESHOLD-$TYPE.pth $METHOD --frequency_threshold $FREQUENCY_THRESHOLD --num-labeled $LABEL --type $TYPE > $OUTPUT_FILE
        done
    done
done
