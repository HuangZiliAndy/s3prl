#!/bin/bash
#SBATCH -A lgarci27 
#SBATCH --job-name=cpu
#SBATCH --partition=shared
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --export=ALL

SDM1_dir=$1
IHM_CLEAN_dir=$2
annotations=$3

# create clean segments
for split in dev test train; do
  sdm1_dir=${SDM1_dir}/${split}
  output_dir=${IHM_CLEAN_dir}/${split}
  python3 downstream/sep_ami/prepare_clean_segs.py ${sdm1_dir} ${output_dir} ${annotations}
  python3 downstream/sep_ami/filter_utt.py ${output_dir} ${output_dir}_filter --min_dur 3.0
done
