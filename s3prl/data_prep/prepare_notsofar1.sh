#!/bin/bash

NSF_dir=/export/fs06/hzili1/datasets/NOTSOFAR1/nsf/
cond=SDM1
output_dir=/export/c02/hzili1/datasets/s3prl_csp/data/NOTSOFAR1

python3 data_prep/prepare_notsofar1.py \
	${NSF_dir} \
	${output_dir} \
	--cond ${cond}
