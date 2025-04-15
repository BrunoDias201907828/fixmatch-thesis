#!/bin/bash

METHOD="FixMatch_Distance"
FREQUENCY_THRESHOLDS="0.6 0.65 0.7 0.75 0.8 0.85"
TYPES="cosine euclidean"

for FREQUENCY_THRESHOLD in $FREQUENCY_THRESHOLDS; do
    for TYPE in $TYPES; do
        OUTPUT_FILE="${METHOD}-${FREQUENCY_THRESHOLD}-${TYPE}.txt"
        echo "Running $METHOD with frequency_threshold=$FREQUENCY_THRESHOLD and type=$TYPE"
        python new_train.py model-$METHOD-$FREQUENCY_THRESHOLD-$TYPE.pth $METHOD --frequency_threshold $FREQUENCY_THRESHOLD --type $TYPE > $OUTPUT_FILE
    done
done
