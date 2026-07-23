#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=15360
#SBATCH --job-name=prepare_asr_seg
#SBATCH --time=1-00:00:00
#SBATCH --exclude=c04,octopod

source path.sh

input_dir=/workspace/dataset/CSPB/data/AMI/MDM
output_dir=/workspace/dataset/CSPB/downstream/diar_ami/MDM

for split in dev test train; do
  python3 downstream/diar_ami/prepare_diar_seg.py \
	--normalize 1 \
	${input_dir}/${split} \
	${output_dir}/${split}
  python3 downstream/diar_ami/filter_seg.py ${output_dir}/${split} ${output_dir}/${split}_filter
done
