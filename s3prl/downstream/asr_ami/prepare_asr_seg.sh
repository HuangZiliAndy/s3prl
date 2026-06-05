#!/bin/bash
#
# prepare_asr_seg.sh — Prepare per-utterance ASR data directories for the AMI corpus.
#
# This script runs in two stages for each recording condition:
#
#   Stage 1 — prepare_asr_seg.py
#     Reads recording-level Kaldi data directories (produced by data_prep/prepare_ami.sh)
#     and cuts each recording into individual utterance WAV files using the segments file.
#     Writes wav.scp, text, utt2spk, reco2dur, and utt2num_samples for each split.
#
#   Stage 2 — filter_utt.py
#     Filters utterances by duration (default: 0.1 s – 20.0 s) for the train and dev splits,
#     writing *_filter directories that are used directly by the training script.
#     test is left unfiltered.
#
# Prerequisites:
#   Run data_prep/prepare_ami.sh first to populate the input Kaldi data directories.
#
# Usage:
#   bash downstream/asr_ami/prepare_asr_seg.sh

source path.sh

# Root directory containing the Kaldi data directories produced by prepare_ami.sh
# Expected structure: ${ami_data_dir}/${cond}/{train,dev,test}/
ami_data_dir=/workspace/dataset/CSPB/data/AMI

# Root output directory for the segmented ASR data
# Outputs will be written to: ${output_base_dir}/${cond}/{train,dev,test}/
#                          and ${output_base_dir}/${cond}/{train_filter,dev_filter}/
output_base_dir=/workspace/dataset/CSPB/downstream/asr_ami

# Utterances outside [min_dur, max_dur] are excluded by prepare_asr_seg.py.
# The actual training duration filter is applied by filter_utt.py (stage 2).
# Set wide bounds here to keep all segments and let filter_utt.py decide.
min_dur=0.0
max_dur=10000.0

# Recording conditions to process. Examples:
#   SDM1 — Single Distant Microphone (mono)
#   MDM  — Multiple Distant Microphones (multi-channel)
for cond in SDM1; do
  input_dir=${ami_data_dir}/${cond}
  output_dir=${output_base_dir}/${cond}

  # Stage 1: cut recordings into per-utterance WAV files
  for split in train dev test; do
    python3 downstream/asr_ami/prepare_asr_seg.py \
        ${input_dir}/${split} \
        ${output_dir}/${split} \
        --min_dur ${min_dur} \
        --max_dur ${max_dur}
  done

  # Stage 2: apply duration filtering to train and dev
  # filter_utt.py defaults: --min_dur 0.1 --max_dur 20.0
  python3 downstream/asr_ami/filter_utt.py ${output_dir}/train ${output_dir}/train_filter
  python3 downstream/asr_ami/filter_utt.py ${output_dir}/dev   ${output_dir}/dev_filter
done
