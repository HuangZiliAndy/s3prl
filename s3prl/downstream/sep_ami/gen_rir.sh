#!/bin/bash
#SBATCH -A lgarci27 
#SBATCH --job-name=cpu
#SBATCH --partition=shared
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --export=ALL

export PATH="/scratch4/lgarci27/hzili1/anaconda3/envs/csp/bin:$PATH"

output_dir=$1

python3 downstream/sep_ami/gen_rir.py $output_dir --num_rirs 50000

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
