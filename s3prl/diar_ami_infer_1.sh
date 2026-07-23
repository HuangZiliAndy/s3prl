#!/bin/bash
#SBATCH --partition=gpu-a100
#SBATCH --account=a100acct
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=15360
#SBATCH --job-name=diar_eval
#SBATCH --time=3-00:00:00
#SBATCH --gpus=1

source path.sh

echo `hostname`

gpus=1
port=25652
normalize=1

for expname in unix_enc_custom_local_cfg20_0.001_mdm_0,4_trainchpos_bf16 unix_enc_custom_local_cfg20_0.0001_mdm_0,4_trainchpos_bf16 unix_enc_custom_local_cfg20_0.001_mdm_0,2,4,6_trainchpos_bf16 unix_enc_custom_local_cfg20_0.0001_mdm_0,2,4,6_trainchpos_bf16; do
  exp_dir="exp/diar_ami/${expname}"
  ckpt="${exp_dir}/best-states-dev.ckpt"

  cond=$(echo "$expname" | sed -n 's/.*_\(mdm_[0-9,]*\)_trainchpos.*/\1/p')
  if [[ "$cond" == "sdm1" ]]; then
      data="SDM1"
      channel="0"
  elif [[ "$cond" == "mdm_bf0,4" ]]; then
      data="MDM_BF0,4"
      channel="0"
  elif [[ "$cond" == "mdm_bf0,2,4,6" ]]; then
      data="MDM_BF0,2,4,6"
      channel="0"
  elif [[ "$cond" == "mdm_bfall" ]]; then
      data="MDM_BF"
      channel="0"
  elif [[ "$cond" == "mdm_0,4" ]]; then
      data="MDM"
      channel="0,4"
  elif [[ "$cond" == "mdm_0,2,4,6" ]]; then
      data="MDM"
      channel="0,2,4,6"
  elif [[ "$cond" == "mdm_all" ]]; then
      data="MDM"
      channel="0,1,2,3,4,5,6,7"
  else
      exit 1;
  fi
  
  data_dir=/workspace/dataset/CSPB/data/AMI/${data}
  dev_dir="${data_dir}/dev"
  test_dir="${data_dir}/test"
  
  echo $dev_dir
  echo $test_dir
  echo $channel
  echo $ckpt
  
  segmentation_thres=0.5
  
  best_der=100
  best_threshold=0
  output_dir=$exp_dir/rttm_gt_spk_assign/test
  
  python3 downstream/diar_ami/evaluate_v1.py \
	$ckpt \
	$test_dir \
	$output_dir \
	--channel $channel \
	--gt_spk_assign 1 \
	--normalize $normalize \
	--segmentation_thres $segmentation_thres \
        --ref_rttm $test_dir/ref_rttm

  cat $output_dir/*.rttm > $output_dir/hyp_rttm
  ./downstream/diar_ami/md-eval.pl -r ${test_dir}/ref_rttm -s $output_dir/hyp_rttm -u ${test_dir}/uem
done
