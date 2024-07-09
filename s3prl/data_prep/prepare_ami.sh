#!/bin/bash

BUT_repo_dir=/scratch4/lgarci27/hzili1/workspace/AMI-diarization-setup
AMI_dir=/data/lgarci27/hzili1/datasets/amicorpus
cond=SDM1
output_dir=/data/lgarci27/hzili1/datasets/s3prl_csp/data/AMI/${cond}_debug
espnet_dir=/scratch4/lgarci27/hzili1/workspace/espnet/egs2/ami/asr1/

# Prepare AMI dataset
python3 data_prep/prepare_ami.py $BUT_repo_dir $AMI_dir $output_dir --cond $cond

# Create segments, utt2spk, text files from ESPnet directory
awk '{split($1, a, "_"); print a[2]"_"a[4]"_"a[5]"_"a[6], a[2], $3, $4}' ${espnet_dir}/data/sdm1_train/segments > ${output_dir}/train/segments 
awk '{split($1, a, "_"); print a[2]"_"a[4]"_"a[5]"_"a[6], a[2], $3, $4}' ${espnet_dir}/data/sdm1_dev/segments > ${output_dir}/dev/segments
awk '{split($1, a, "_"); print a[2]"_"a[4]"_"a[5]"_"a[6], a[2], $3, $4}' ${espnet_dir}/data/sdm1_eval/segments > ${output_dir}/test/segments

awk -F' ' '{split($1, a, "_"); print $1, a[1]}' ${output_dir}/train/segments > ${output_dir}/train/utt2spk
awk -F' ' '{split($1, a, "_"); print $1, a[1]}' ${output_dir}/dev/segments > ${output_dir}/dev/utt2spk
awk -F' ' '{split($1, a, "_"); print $1, a[1]}' ${output_dir}/test/segments > ${output_dir}/test/utt2spk

awk '{split($1, a, "_"); $1=a[2]"_"a[4]"_"a[5]"_"a[6]; print $0}' ${espnet_dir}/data/sdm1_train/text > ${output_dir}/train/text
awk '{split($1, a, "_"); $1=a[2]"_"a[4]"_"a[5]"_"a[6]; print $0}' ${espnet_dir}/data/sdm1_dev/text > ${output_dir}/dev/text
awk '{split($1, a, "_"); $1=a[2]"_"a[4]"_"a[5]"_"a[6]; print $0}' ${espnet_dir}/data/sdm1_eval/text > ${output_dir}/test/text
