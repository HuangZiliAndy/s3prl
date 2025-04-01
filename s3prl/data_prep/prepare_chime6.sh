#!/bin/bash

CHiME6_dir=/export/corpora6/CHiME6
cond=SDM1
output_dir=/export/c02/hzili1/datasets/s3prl_csp/data/CHiME6

python3 data_prep/prepare_chime6.py \
	${CHiME6_dir} \
	${output_dir} \
	--cond ${cond}
