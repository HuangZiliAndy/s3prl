#!/bin/bash
#SBATCH -A lgarci27
#SBATCH --job-name=prepare_asr_seg
#SBATCH --partition=shared
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --export=ALL

export PATH="/scratch4/lgarci27/hzili1/anaconda3/envs/csp/bin:$PATH"

min_dur=0.0
max_dur=10000.0
input_dir=/data/lgarci27/hzili1/datasets/s3prl_csp/data/AMI/SDM1
output_dir=/scratch4/lgarci27/hzili1/datasets/s3prl_csp/s3prl/downstream/asr_ami/SDM1

for split in dev test train; do
  python downstream/asr_ami/prepare_asr_seg.py \
      ${input_dir}/${split} \
      ${output_dir}/${split} \
      --min_dur $min_dur --max_dur $max_dur
  python3 downstream/asr_ami/filter_utt.py ${output_dir}/${split} ${output_dir}/${split}_filter
done
