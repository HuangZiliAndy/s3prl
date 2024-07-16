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

upstream=dac_16kHz
lr=0.001
exp_dir=/export/c12/hzili1/workspace/JSALT24/workspace/s3prl/s3prl/exp_new/ctc/${upstream}_${lr}

python3 run_downstream.py \
	-p ${exp_dir} \
	-m train \
	-u ${upstream} \
	-d ctc \
	-c downstream/ctc/libriphone.yaml \
	-o "config.optimizer.lr=${lr},,config.downstream_expert.corpus.num_workers=12,,config.runner.gradient_accumulate_steps=1,,config.downstream_expert.corpus.batch_size=32"
