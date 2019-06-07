#!/bin/bash

python ParlAI/parlai/scripts/interactive.py \
    -t parlai.agents.local_human.local_human:LocalHumanAgent \
    -m parlai_external.agents.seq2seq_with_partner.seq2seq_with_partner:Seq2seqPartnerAgent \
    -bs 1 \
    -dp ./data/ \
    --model-file ./models/self_model \
    --dict-file ./models/convai2.dict \
    --init-partner-model ./models/partner_mindless_model \
    --skip-generation False \
    --mtbeam 1 \
    --beam-size 5 \
    --beam-num-iterations 2 \
    --beam-block-ngram 3 \
    --no-cuda \
    --beam-dist-threshold 3 \
    --egocentric \
