#!/bin/bash

MMCSG_dir=/export/corpora7/Dipco
cond=SDM1
output_dir=/export/fs05/hzili1/datasets/s3prl_csp/data/Dipco

python3 data_prep/prepare_dipco.py \
	${MMCSG_dir} \
	${output_dir} \
	--cond ${cond} \
	--merge_dis 0.0
