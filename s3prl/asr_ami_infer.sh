#!/bin/bash
#SBATCH -A lgarci27_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10240
#SBATCH --partition=a100
#SBATCH --job-name=asr_ami
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH -G 1
#SBATCH --exclude=gpu12

export PATH="/scratch4/lgarci27/hzili1/anaconda3/envs/s3prl_csp/bin:$PATH"

gpus=1
port=25678

exp_dir="/data/lgarci27/hzili1/workspace/s3prl_csp/s3prl/exp/asr_ami/wavlm_base_plus_0.0001/"
ckpt="${exp_dir}/dev-best.ckpt"
test_dir="/scratch4/lgarci27/hzili1/datasets/s3prl_csp/s3prl/downstream/asr_ami/SDM1/test"

python3 run_downstream.py \
    -m evaluate \
    -e $ckpt \
    -o "config.downstream_expert.datarc.max_samples=1000000,,config.downstream_expert.loaderrc.eval_batchsize=1,,config.downstream_expert.loaderrc.test_dir=${test_dir},,config.downstream_expert.datarc.mch=False,,config.downstream_expert.datarc.channel='0'"

./downstream/asr_ami/score.sh $exp_dir false $test_dir

python3 run_downstream.py \
	-m evaluate \
	-e $ckpt \
	-o "config.downstream_expert.datarc.max_samples=1000000,,config.downstream_expert.loaderrc.eval_batchsize=1,,config.downstream_expert.loaderrc.test_dir=${test_dir},,config.downstream_expert.datarc.mch=False,,config.downstream_expert.datarc.channel='0',,config.downstream_expert.datarc.decoder_args.decoder_type='kenlm'"

./downstream/asr_ami/score.sh $exp_dir true $test_dir
