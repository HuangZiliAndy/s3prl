#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=15360
#SBATCH --job-name=asr_ami
#SBATCH --time=3-00:00:00
#SBATCH --gpus=1
#SBATCH --exclude=c04

source path.sh

gpus=1
port=25678

test_dir="/workspace/dataset/CSPB/downstream/asr_ami/MDM/test"

for entry in \
    "unix_enc_custom_local_cfg20_0.0001_mdm_0,4_trainchpos_bf16:0,4" \
    "unix_enc_custom_local_cfg20_0.0001_mdm_0,2,4,6_trainchpos_bf16:0,2,4,6"
do
    exp_name="${entry%%:*}"
    channel="${entry##*:}"
    exp_dir="/workspace/workspace/s3prl/s3prl/exp/asr_ami/${exp_name}/"
    ckpt="${exp_dir}/dev-best.ckpt"

    python3 run_downstream.py \
        -m evaluate \
        -e $ckpt \
        -o "config.downstream_expert.datarc.max_samples=1000000,,config.downstream_expert.loaderrc.eval_batchsize=1,,config.downstream_expert.loaderrc.test_dir=${test_dir},,config.downstream_expert.datarc.channel='${channel}'"

    ./downstream/asr_ami/score.sh $exp_dir false $test_dir
done
