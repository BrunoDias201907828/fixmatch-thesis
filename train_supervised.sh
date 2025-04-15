#!/bin/bash

METHOD="Supervised"
LABELS="250 4000 10000 20000 40000"

for LABEL in $LABELS; do
    OUTPUT_FILE="${METHOD}-${LABEL}.txt"
    echo "Running $METHOD with labeled samples=$LABEL"
    python new_train.py model-$METHOD-$LABEL.pth $METHOD --num-labeled $LABEL > $OUTPUT_FILE
done