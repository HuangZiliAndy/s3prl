#!/bin/bash

#SBATCH --job-name=s3prl_pr
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=12
#SBATCH --gpus=1 
#SBATCH --partition=gpu-a100 
#SBATCH --time=3-00:00:00
#SBATCH --mem=20480
#SBATCH --exclude=e04
#SBATCH --account=a100acct

module load cuda/12.1
export PATH="/home/hzili1/anaconda3/envs/s3prl/bin:$PATH"

for lr in 0.01 0.001; do
  upstream=dac_16kHz_no_quantize
  exp_dir=/export/c12/hzili1/workspace/JSALT24/workspace/s3prl/s3prl/exp/voxceleb1/${upstream}_${lr}
  
  echo $exp_dir
  python3 run_downstream.py -m evaluate -e ${exp_dir}/dev-best.ckpt
done
