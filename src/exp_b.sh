#!/bin/bash

fairseq-eval-lm \
    /data/text/ \
    --path /data/models/ba3072seed3.pt \
    --sample-break-mode none \
    --gen-subset valid \
    --max-sentences 1 \
    --context-window 2560