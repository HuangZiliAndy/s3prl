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
module load sox
export PATH="/home/hzili1/anaconda3/envs/s3prl/bin:$PATH"

upstream=dac_local

for model_type in l1 l1_cos log_coss cos; do 
  upstream_path=/export/fs06/alaurent1/${model_type}/best/dackd/weights.pth
  
  for lr in 0.001; do
    exp_dir=/export/c12/hzili1/workspace/JSALT24/workspace/s3prl/s3prl/exp_new/ctc/DACKD_${model_type}_best_${lr}_mini
    
    python3 run_downstream.py \
    	-p ${exp_dir} \
    	-m train \
    	-u ${upstream} \
  	-g "320,DACKD,z_q" \
  	-k "${upstream_path}" \
    	-d ctc \
    	-c downstream/ctc/libriphone_mini.yaml \
    	-o "config.optimizer.lr=${lr},,config.downstream_expert.corpus.num_workers=12,,config.runner.gradient_accumulate_steps=1,,config.downstream_expert.corpus.batch_size=32"
  done
done
