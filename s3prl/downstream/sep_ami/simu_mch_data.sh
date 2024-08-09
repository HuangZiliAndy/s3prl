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
export PYTHONPATH="/scratch4/lgarci27/hzili1/workspace/CSP_publish/s3prl:$PYTHONPATH"

wham_noise_dir=/data/lgarci27/hzili1/datasets/wham_noise

RIR_dir=/data/lgarci27/hzili1/workspace/s3prl_csp/s3prl/downstream/sep_ami/data/ami/AMI_RIRs_3srcs # simulated with downstream/sep_ami/gen_rir.sh 
#./downstream/sep_ami/gen_rir.sh $RIR_dir

SDM1_dir=/data/lgarci27/hzili1/datasets/s3prl_csp/data/AMI/SDM1
#IHM_CLEAN_dir=/data/lgarci27/hzili1/datasets/s3prl_csp/s3prl/downstream/sep_ami/IHM_CLEAN
IHM_CLEAN_dir=/data/lgarci27/hzili1/datasets/s3prl_csp/s3prl/debug/IHM_CLEAN
annotations=/data/lgarci27/hzili1/datasets/ami_public_manual_1.6.2
./downstream/sep_ami/prepare_clean_segs.sh $SDM1_dir $IHM_CLEAN_dir $annotations 

#for split in dev test train; do
#  data_dir="/data/lgarci27/hzili1/datasets/s3prl_csp/s3prl/downstream/sep_ami/IHM_CLEAN/${split}_filter"
#  RIR_dir="${RIR_dir}/${split}"
#  num_spk=1,2
#  num_spk_prob=0.5,0.5
#  add_noise=1
#  add_reverb=1
#  sir_range="5,20"
#  snr_range="5,20"
#  normalize=1
#  s1_first=0
#  s1_only=1
#  full_overlap=0
#  seed=7
#  noise_type="none"
#  single_channel=1
#  output_dir="/scratch4/lgarci27/hzili1/workspace/Amazon/fairseq/examples/enhancement/${split}"
#  
#  if [ "$split" == "train" ]; then
#    noise_scp_file="${wham_noise_dir}/tr.scp"
#    num_utts=20000
#  fi
#  if [ "$split" == "dev" ]; then
#    noise_scp_file="${wham_noise_dir}/cv.scp"
#    num_utts=1000
#  fi
#  if [ "$split" == "test" ]; then
#    noise_scp_file="${wham_noise_dir}/tt.scp"
#    num_utts=1000
#  fi
#  
#  python downstream/sep_ami/simu_mch_data.py \
#  	$data_dir \
#  	$output_dir \
#  	--noise_scp_file $noise_scp_file \
#	--RIR_dir $RIR_dir \
#  	--num_spk $num_spk \
#  	--num_spk_prob $num_spk_prob \
#  	--add_noise $add_noise \
#	--noise_type $noise_type \
#  	--add_reverb $add_reverb \
#  	--sir_range $sir_range \
#  	--snr_range $snr_range \
#  	--num_utts $num_utts \
#  	--normalize $normalize \
#  	--s1_first $s1_first \
#	--s1_only $s1_only \
#  	--full_overlap $full_overlap \
#    --single_channel $single_channel \
#  	--seed $seed 
#done
