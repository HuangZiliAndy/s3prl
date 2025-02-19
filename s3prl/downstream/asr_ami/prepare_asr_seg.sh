#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=15360
#SBATCH --job-name=prepare_asr_seg
#SBATCH --time=1-00:00:00

export PATH="/home/hzili1/anaconda3/envs/s3prl_csp/bin:$PATH"

min_dur=0.0
max_dur=10000.0

#input_dir=/export/c02/hzili1/datasets/s3prl_csp/data/AMI/SDM1
#output_dir=/export/c02/hzili1/datasets/s3prl_csp/downstream/asr_ami/SDM1

#input_dir=/export/c02/hzili1/datasets/s3prl_csp/data/AMI/MDM
#output_dir=/export/c02/hzili1/datasets/s3prl_csp/downstream/asr_ami/MDM

input_dir=/export/c02/hzili1/datasets/s3prl_csp/data/AMI/MDM_BF
output_dir=/export/c02/hzili1/datasets/s3prl_csp/downstream/asr_ami/MDM_BF

for split in dev test train; do
  python downstream/asr_ami/prepare_asr_seg.py \
      ${input_dir}/${split} \
      ${output_dir}/${split} \
      --min_dur $min_dur --max_dur $max_dur
  python3 downstream/asr_ami/filter_utt.py ${output_dir}/${split} ${output_dir}/${split}_filter
done
