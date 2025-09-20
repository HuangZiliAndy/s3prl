#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=15360
#SBATCH --job-name=prepare_asr_seg
#SBATCH --time=1-00:00:00
#SBATCH --exclude=c04,octopod

export PATH="/export/c02/hzili1/tmp/home/hzili1/anaconda3/envs/csp/bin:$PATH"

#input_dir=/export/c02/hzili1/datasets/s3prl_csp/data/AMI/SDM1
#output_dir=/export/c02/hzili1/datasets/s3prl_csp/downstream/diar_ami/SDM1

#input_dir=/export/c02/hzili1/datasets/s3prl_csp/data/AMI/MDM
#output_dir=/export/c02/hzili1/datasets/s3prl_csp/downstream/diar_ami/MDM

#input_dir=/export/c02/hzili1/datasets/s3prl_csp/data/AMI/MDM_BF
#output_dir=/export/c02/hzili1/datasets/s3prl_csp/downstream/diar_ami/MDM_BF

#input_dir=/export/c02/hzili1/datasets/s3prl_csp/data/AMI/MDM_BF0,4
#output_dir=/export/c02/hzili1/datasets/s3prl_csp/downstream/diar_ami/MDM_BF0,4

#input_dir=/export/c02/hzili1/datasets/s3prl_csp/data/AMI/MDM_BF0,2,4,6
#output_dir=/export/c02/hzili1/datasets/s3prl_csp/downstream/diar_ami/MDM_BF0,2,4,6

#input_dir=/export/c02/hzili1/datasets/s3prl_csp/data/AMI/sep_cfg0_SoudenMVDR_2CH
#output_dir=/export/fs05/hzili1/datasets/s3prl_csp/downstream/diar_ami/sep_cfg0_SoudenMVDR_2CH

#input_dir=/export/c02/hzili1/datasets/s3prl_csp/data/AMI/sep_cfg0_SoudenMVDR_4CH
#output_dir=/export/fs05/hzili1/datasets/s3prl_csp/downstream/diar_ami/sep_cfg0_SoudenMVDR_4CH

#input_dir=/export/c02/hzili1/datasets/s3prl_csp/data/AMI/sep_cfg0_SoudenMVDR_8CH
#output_dir=/export/fs05/hzili1/datasets/s3prl_csp/downstream/diar_ami/sep_cfg0_SoudenMVDR_8CH

for split in dev test train; do
  python3 downstream/diar_ami/prepare_diar_seg.py \
	--normalize 1 \
	${input_dir}/${split} \
	${output_dir}/${split}
  python3 downstream/diar_ami/filter_seg.py ${output_dir}/${split} ${output_dir}/${split}_filter
done
