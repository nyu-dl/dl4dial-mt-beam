#!/bin/bash

python ParlAI/parlai/scripts/eval_model.py \
    -dt valid \
    -t convai2:self \
    -m parlai_external.agents.seq2seq_with_partner.seq2seq_with_partner:Seq2seqPartnerAgent \
    -bs 64 \
    -dp ./data/ \
    --model-file ./models/self_model \
    --dict-file ./models/convai2.dict \
    --init-partner-model ./models/partner_mindless_model \
    --skip-generation False \
    --mtbeam 1 \
    --beam-size 5 \
    --beam-num-iterations 4 \
    --beam-block-ngram 3 \
    --beam-dist-threshold 3 \
    --no-cuda \
