#!/bin/bash
#SBATCH -A lgarci27_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=15360
#SBATCH --partition=a100
#SBATCH --job-name=asr_ami
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:1
#SBATCH -G 1
#SBATCH --exclude=gpu12

export PATH="/scratch4/lgarci27/hzili1/anaconda3/envs/s3prl_csp/bin:$PATH"
export PYTHONPATH="/scratch4/lgarci27/hzili1/workspace/CSP_publish/s3prl:$PYTHONPATH"

upstream=wavlm_base_plus
lr=0.0001
gpus=1
port=25652
#distributed="-m torch.distributed.launch --nproc_per_node ${gpus} --master_port ${port}"
distributed=""
exp_dir="`pwd`/exp/asr_ami/${upstream}_${lr}"

python3 $distributed run_downstream.py \
    -p $exp_dir \
    -m train \
    -u $upstream \
    -d asr_ami \
    -c downstream/asr_ami/config/AMI/cfg.yaml \
    -o "config.runner.gradient_accumulate_steps=1,,config.downstream_expert.datarc.channel='0',,config.optimizer.lr=${lr}"
