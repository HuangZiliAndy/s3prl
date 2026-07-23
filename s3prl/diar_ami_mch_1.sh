#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=15360
#SBATCH --job-name=diar_ami
#SBATCH --time=5-00:00:00
#SBATCH --gpus=1

source path.sh

# Directory where s3prl caches upstream model weights
cache_dir=/workspace/downloads/s3prl

upstream=unix_enc_custom_local
ckpt=/workspace/workspace/MelHuBERT/exp/cfg20/checkpoint-200000
cfg_name=$(basename $(dirname ${ckpt}))

gpus=1
port=25652
#distributed="-m torch.distributed.launch --nproc_per_node ${gpus} --master_port ${port}"
distributed=""

cond=mdm_0,2,4,6
data="MDM"
channel="0,2,4,6"

# Root directory containing the diarization data produced by data_prep/prepare_ami.sh.
# Expected sub-directories: ${data}/train_filter, ${data}/dev_filter
diar_data_dir=/workspace/dataset/CSPB/downstream/diar_ami
train_dir=${diar_data_dir}/${data}/train_filter
dev_dir=${diar_data_dir}/${data}/dev_filter

# Run training for each learning rate (hyperparameter search).
for lr in 0.001 0.0001; do
  exp_dir="`pwd`/exp/diar_ami/${upstream}_${cfg_name}_${lr}_${cond}_trainchpos_bf16"
  echo ${train_dir}
  echo ${dev_dir}
  echo ${channel}
  echo ${exp_dir}

  python3 $distributed run_downstream.py \
      --cache_dir ${cache_dir} \
      -p $exp_dir \
      -m train \
      -u $upstream \
      -k $ckpt \
      --upstream_trainable --train_channel_pos_only 1 \
      -d diar_ami \
      -c downstream/diar_ami/config/AMI/cfg.yaml \
      -o "config.downstream_expert.datarc.channel='${channel}',,config.optimizer.lr=${lr},,config.downstream_expert.loaderrc.train_dir=${train_dir},,config.downstream_expert.loaderrc.dev_dir=${dev_dir},,config.runner.fp16=False,,config.runner.bf16=True"
done
