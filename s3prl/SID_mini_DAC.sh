#!/bin/bash

#SBATCH --job-name=s3prl_sid
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=12
#SBATCH --gpus=1 
#SBATCH --partition=gpu-a100 
#SBATCH --time=3-00:00:00
#SBATCH --mem=30720
#SBATCH --exclude=e04
#SBATCH --account=a100acct

module load cuda/12.1
export PATH="/home/hzili1/anaconda3/envs/s3prl/bin:$PATH"

upstream=dac_local

for lr in 0.01 0.001 0.0001; do
  exp_dir=/export/c12/hzili1/workspace/JSALT24/workspace/s3prl/s3prl/exp_new/voxceleb1/DACKD_l1_best_${lr}_mini
  
  python3 run_downstream.py \
  	-p ${exp_dir} \
  	-m train \
  	-u ${upstream} \
	-g "320,DACKD,z_q" \
	-k "/export/fs06/alaurent1/l1/best/dackd/weights.pth" \
  	-d voxceleb1 \
  	-c downstream/voxceleb1/config_mini.yaml \
  	-o "config.optimizer.lr=${lr},,config.downstream_expert.datarc.num_workers=12,,config.runner.gradient_accumulate_steps=1,,config.downstream_expert.datarc.train_batch_size=32"
done
