#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=15360
#SBATCH --job-name=beamformit
#SBATCH --time=3-00:00:00
#SBATCH --exclude=c04

export BEAMFORMIT=/export/c02/hzili1/workspace/espnet/tools/BeamformIt
export PATH=${PATH}:${BEAMFORMIT}
export PATH="/export/c02/hzili1/tmp/home/hzili1/anaconda3/envs/csp/bin:$PATH"

#mdm_dir=/export/c02/hzili1/workspace/multi-channel/mch_dataset/
#mdm_bf_dir=/export/c02/hzili1/workspace/multi-channel/mch_bf_dataset
#nj=16
#
#for dset in AISHELL-4 Alimeeting Alimeeting_Eval AMI AMI_dev CHiME6 ICSI NOTSOFAR1 NOTSOFAR1_dev; do 
#  python3 data_prep/beamformit_kaldi_v1.py ${mdm_dir}/${dset}/wav.scp ${mdm_bf_dir}/${dset} --config_file data_prep/beamformit.cfg --num_jobs ${nj}
#done

#mdm_dir=/export/c02/hzili1/datasets/s3prl_csp/data/AMI/MDM
#mdm_bf_dir=/export/c02/hzili1/datasets/s3prl_csp/data/AMI/MDM_BF0,4
#nj=16
#
#for dset in test dev train; do 
#  python3 data_prep/beamformit_kaldi_v1.py ${mdm_dir}/${dset}/wav.scp ${mdm_bf_dir}/${dset} --config_file data_prep/beamformit.cfg --num_jobs ${nj} --channels 0,4
#done
#
#mdm_dir=/export/c02/hzili1/datasets/s3prl_csp/data/AMI/MDM
#mdm_bf_dir=/export/c02/hzili1/datasets/s3prl_csp/data/AMI/MDM_BF0,2,4,6
#nj=16
#
#for dset in test dev train; do 
#  python3 data_prep/beamformit_kaldi_v1.py ${mdm_dir}/${dset}/wav.scp ${mdm_bf_dir}/${dset} --config_file data_prep/beamformit.cfg --num_jobs ${nj} --channels 0,2,4,6
#done

#mdm_dir=/export/c02/hzili1/datasets/s3prl_csp/data/Alimeeting/MDM
#mdm_bf_dir=/export/c02/hzili1/datasets/s3prl_csp/data/Alimeeting/MDM_BF0,4
#nj=16
#
#for dset in Eval Test Train; do 
#  python3 data_prep/beamformit_kaldi_v1.py ${mdm_dir}/${dset}/wav.scp ${mdm_bf_dir}/${dset} --config_file data_prep/beamformit.cfg --num_jobs ${nj} --channels 0,4
#done
#
#mdm_dir=/export/c02/hzili1/datasets/s3prl_csp/data/Alimeeting/MDM
#mdm_bf_dir=/export/c02/hzili1/datasets/s3prl_csp/data/Alimeeting/MDM_BF0,2,4,6
#nj=16
#
#for dset in Eval Test Train; do 
#  python3 data_prep/beamformit_kaldi_v1.py ${mdm_dir}/${dset}/wav.scp ${mdm_bf_dir}/${dset} --config_file data_prep/beamformit.cfg --num_jobs ${nj} --channels 0,2,4,6
#done

#mdm_dir=/export/c02/hzili1/datasets/s3prl_csp/downstream/asr_ami/MDM
#mdm_bf_dir=/export/c02/hzili1/datasets/s3prl_csp/downstream/asr_ami/MDM_BF0,2,4,6
#nj=16
#
#for dset in dev_filter test train_filter; do 
#  python3 data_prep/beamformit_kaldi_v1.py ${mdm_dir}/${dset}/wav.scp ${mdm_bf_dir}/${dset} --config_file data_prep/beamformit.cfg --num_jobs ${nj} --channels 0,2,4,6
#done

mdm_dir=/export/c02/hzili1/datasets/s3prl_csp/downstream/sep_ami/2spk_reverb_diffuse/MDM
mdm_bf_dir=/export/c02/hzili1/datasets/s3prl_csp/downstream/sep_ami/2spk_reverb_diffuse/MDM_BF0,4
nj=16

for dset in dev test train; do 
  python3 data_prep/beamformit_kaldi_v1.py ${mdm_dir}/${dset}/wav.scp ${mdm_bf_dir}/${dset} --config_file data_prep/beamformit.cfg --num_jobs ${nj} --channels 0,4
done

mdm_dir=/export/c02/hzili1/datasets/s3prl_csp/downstream/sep_ami/2spk_reverb_diffuse/MDM
mdm_bf_dir=/export/c02/hzili1/datasets/s3prl_csp/downstream/sep_ami/2spk_reverb_diffuse/MDM_BF0,2,4,6
nj=16

for dset in dev test train; do 
  python3 data_prep/beamformit_kaldi_v1.py ${mdm_dir}/${dset}/wav.scp ${mdm_bf_dir}/${dset} --config_file data_prep/beamformit.cfg --num_jobs ${nj} --channels 0,2,4,6
done

mdm_dir=/export/c02/hzili1/datasets/s3prl_csp/downstream/sep_ami/2spk_reverb_diffuse/MDM
mdm_bf_dir=/export/c02/hzili1/datasets/s3prl_csp/downstream/sep_ami/2spk_reverb_diffuse/MDM_BF
nj=16

for dset in dev test train; do 
  python3 data_prep/beamformit_kaldi_v1.py ${mdm_dir}/${dset}/wav.scp ${mdm_bf_dir}/${dset} --config_file data_prep/beamformit.cfg --num_jobs ${nj} --channels 0,1,2,3,4,5,6,7
done
