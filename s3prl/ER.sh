#!/bin/bash

#SBATCH --job-name=s3prl_emotion
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=6
#SBATCH --gpus=1 
#SBATCH --partition=gpu-a100 
#SBATCH --time=3-00:00:00
#SBATCH --mem=20480
#SBATCH --exclude=e04
#SBATCH --account=a100acct

module load cuda/12.1
export PATH="/home/hzili1/anaconda3/envs/s3prl/bin:$PATH"

upstream=fbank
lr=0.001
exp_dir=/export/c12/hzili1/workspace/JSALT24/workspace/s3prl/s3prl/exp_new/emotion/${upstream}_${lr}

for test_fold in fold1 fold2 fold3 fold4 fold5;
do
    python3 run_downstream.py \
	    -p ${exp_dir}_${test_fold} \
	    -m train \
	    -u ${upstream} \
	    -d emotion \
	    -c downstream/emotion/config.yaml \
	    -o "config.optimizer.lr=${lr},,config.downstream_expert.datarc.test_fold='$test_fold'"
    python3 run_downstream.py -m evaluate -e ${exp_dir}_${test_fold}/dev-best.ckpt
done
