#!/bin/bash

Alimeeting_dir=/export/c02/hzili1/datasets/Alimeeting
cond=SDM1
output_dir=/export/c02/hzili1/datasets/s3prl_csp/data/Alimeeting

python3 data_prep/prepare_alimeeting.py \
	${Alimeeting_dir} \
	${output_dir} \
	--cond ${cond}
