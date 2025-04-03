#!/bin/bash
METHODS="FixMatch_DeepBilevel"
for METHOD in $METHODS; do
    echo "train $METHOD"
    python train.py model-$METHOD.pth $METHOD
done
