#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=15360
#SBATCH --job-name=beamformit
#SBATCH --time=3-00:00:00

export BEAMFORMIT=/export/c02/hzili1/workspace/espnet/tools/BeamformIt
export PATH=${PATH}:${BEAMFORMIT}
export PATH="/export/c02/hzili1/tmp/home/hzili1/anaconda3/envs/csp/bin:$PATH"

channel=0,1,2,3,4,5,6,7
mdm_dir=/export/c02/hzili1/datasets/s3prl_csp/data/AMI/MDM
mdm_bf_dir=/export/c02/hzili1/datasets/s3prl_csp/data/AMI/MDM_BF

for dset in dev; do
  python3 data_prep/beamformit_kaldi.py \
	  ${mdm_dir}/${dset}/wav.scp \
	  ${mdm_bf_dir}/${dset} \
	  ${channel} \
	  --config_file data_prep/beamformit.cfg
done
