#!/bin/bash

METHOD="FixMatch_Mcdropout"
CONFIDENCE_THRESHOLDS="0.85 0.9 0.95"
MI_THRESHOLDS="0.1 0.15 0.2 0.25 0.3"
MC_DROPOUT_PASSES="10 30"

for CONFIDENCE_THRESHOLD in $CONFIDENCE_THRESHOLDS; do
    for MI_THRESHOLD in $MI_THRESHOLDS; do
        for MC_DROPOUT_PASS in $MC_DROPOUT_PASSES; do
            OUTPUT_FILE="${METHOD}-${CONFIDENCE_THRESHOLD}-${MI_THRESHOLD}-${MC_DROPOUT_PASS}.txt"
            echo "Running $METHOD with confidence_threshold=$CONFIDENCE_THRESHOLD, mi_threshold=$MI_THRESHOLD, mc_dropout_passes=$MC_DROPOUT_PASS"
            python train.py model-$METHOD.pth $METHOD --confidence_threshold $CONFIDENCE_THRESHOLD --mi_threshold $MI_THRESHOLD --mc_dropout_passes $MC_DROPOUT_PASS > $OUTPUT_FILE
        done
    done
done