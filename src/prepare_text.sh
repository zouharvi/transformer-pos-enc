#!/bin/bash

fairseq-preprocess \
    --only-source \
    --testpref $1 \
    --srcdict /data/wikitext/dict.txt \
    --destdir /data/text/ \
    --workers 20
    # --trainpref /data/wikitext/wiki.train.tokens \