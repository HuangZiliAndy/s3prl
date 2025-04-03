#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=15360
#SBATCH --job-name=asr_ami
#SBATCH --time=3-00:00:00
#SBATCH --gpus=1

source path.sh

mic_arch=AMI
output_dir=/export/c02/hzili1/workspace/s3prl/s3prl/downstream/sep_alimeeting/${mic_arch}_RIRs_3srcs

python3 downstream/sep_ami/gen_rir.py $output_dir --mic_arch ${mic_arch} --num_rirs 50000

mkdir -p ${output_dir}/train
for i in $(seq -f "%07g" 1 30000); do
  mv "${output_dir}/${i}.npz" ${output_dir}/train/.
done

mkdir -p ${output_dir}/dev
for i in $(seq -f "%07g" 30001 40000); do
  mv "${output_dir}/${i}.npz" ${output_dir}/dev/.
done

mkdir -p ${output_dir}/test
for i in $(seq -f "%07g" 40001 50000); do
  mv "${output_dir}/${i}.npz" ${output_dir}/test/.
done
