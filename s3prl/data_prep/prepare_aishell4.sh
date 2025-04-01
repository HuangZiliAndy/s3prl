#!/bin/bash

AISHELL_dir=/export/c02/hzili1/datasets/AISHELL-4
cond=SDM1
output_dir=/export/c02/hzili1/datasets/s3prl_csp/data/AISHELL-4

python3 data_prep/prepare_aishell4.py \
	${AISHELL_dir} \
	${output_dir} \
	--cond ${cond}
