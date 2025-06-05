#!/bin/bash

METHOD="Supervised"
LABELS="39999"

for LABEL in $LABELS; do
    OUTPUT_FILE="${METHOD}-${LABEL}.txt"
    echo "Running $METHOD with labeled samples=$LABEL"
    python supervised_train.py model-$METHOD-$LABEL.pth $METHOD --num-labeled $LABEL > $OUTPUT_FILE
done