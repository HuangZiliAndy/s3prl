#!/bin/bash

ICSI_dir=/export/corpora5/LDC/LDC2004S02/meeting_speech
cond=SDM1
output_dir=/export/c02/hzili1/datasets/s3prl_csp/data/ICSI

python3 data_prep/prepare_icsi.py \
	${ICSI_dir} \
	${output_dir} \
	--cond ${cond}
