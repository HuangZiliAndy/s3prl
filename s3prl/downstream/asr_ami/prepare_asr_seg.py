"""
prepare_asr_seg.py — Cut recording-level WAV files into per-utterance segments
for downstream ASR training.

Reads a Kaldi-style data directory (wav.scp, text, segments) and writes a new
data directory where each utterance is saved as its own WAV file. The output
directory contains wav.scp, text, utt2spk, reco2dur, and utt2num_samples.

Utterances outside [min_dur, max_dur] seconds or with empty transcripts are
skipped. All input audio is assumed to be 16 kHz; an assertion enforces this.

Usage:
  python3 prepare_asr_seg.py <input_dir> <output_dir> [--min_dur 0.1] [--max_dur 20.0]
"""

import io
import os
import argparse
import shutil
import soundfile as sf
import numpy as np
import subprocess

parser = argparse.ArgumentParser(description='Create ASR segments')
parser.add_argument('input_dir', type=str, help='Input directory')
parser.add_argument('output_dir', type=str, help='Output directory')
parser.add_argument('--min_dur', type=float, default=0.1, help='minimum duration')
parser.add_argument('--max_dur', type=float, default=20.0, help='maximum duration')
args = parser.parse_args()

def get_wav_scp(fname):
    """Load a Kaldi wav.scp file into a dict.

    Args:
        fname: Path to wav.scp.

    Returns:
        Dict mapping utterance/recording ID → wav file path (or pipe command).
    """
    utt2path = {}
    with open(fname, 'r') as fh:
        content = fh.readlines()
    for line in content:
        line = line.strip('\n')
        utt, path = line.split(None, 1)
        utt2path[utt] = path
    return utt2path


def get_utt2spk(fname):
    """Load a Kaldi utt2spk file into a dict.

    Args:
        fname: Path to utt2spk.

    Returns:
        Dict mapping segment ID → speaker ID.
    """
    seg2spk = {}
    with open(fname, 'r') as fh:
        content = fh.readlines()
    for line in content:
        line = line.strip('\n')
        seg, spk = line.split()
        seg2spk[seg] = spk
    return seg2spk


def get_text(fname):
    """Load a Kaldi text file into a dict.

    Handles utterances with empty transcripts (lines with only an ID) by
    mapping them to an empty string.

    Args:
        fname: Path to the text file.

    Returns:
        Dict mapping segment ID → transcript string (may be empty).
    """
    seg2text = {}
    with open(fname, 'r') as fh:
        content = fh.readlines()
    for line in content:
        line = line.strip('\n')
        try:
            seg, text = line.split(None, 1)
        except ValueError:
            seg = line
            text = ""
        seg2text[seg] = text
    return seg2text


def get_segments(fname):
    """Load a Kaldi segments file grouped by recording ID.

    Each line has: segment_id  recording_id  start_time  end_time

    Args:
        fname: Path to the segments file.

    Returns:
        Dict mapping recording ID → list of [segment_id, start_t, end_t].
    """
    utt2segs = {}
    with open(fname, 'r') as fh:
        content = fh.readlines()
    for line in content:
        line = line.strip('\n')
        seg, utt, start_t, end_t = line.split()
        start_t, end_t = round(float(start_t), 2), round(float(end_t), 2)
        if utt not in utt2segs:
            utt2segs[utt] = []
        utt2segs[utt].append([seg, start_t, end_t])
    return utt2segs

def main():
    """Cut recording-level WAVs into per-segment WAV files and write Kaldi output files.

    For each recording in wav.scp:
      - Reads the audio (WAV file or sox/pipe command ending in '|').
      - Asserts 16 kHz sample rate.
      - For each segment in the segments file, slices the audio array and writes
        a new WAV file, skipping segments that are too short/long or have empty text.

    Output files written to output_dir/:
      wav.scp, text, utt2spk, reco2dur, utt2num_samples
    """
    if not os.path.exists(args.output_dir + '/wav'):
        os.makedirs(args.output_dir + '/wav')
    utt2path = get_wav_scp("{}/wav.scp".format(args.input_dir))
    seg2text = get_text("{}/text".format(args.input_dir))
    utt2segs = get_segments("{}/segments".format(args.input_dir))
    for utt in utt2segs:
        if utt not in utt2path:
            print("{} not in wav.scp, skipping it".format(utt))
    uttlist = list(utt2path.keys())
    uttlist.sort()

    wav_scp_file = open("{}/wav.scp".format(args.output_dir), 'w')
    text_file = open("{}/text".format(args.output_dir), 'w')
    utt2spk_file = open("{}/utt2spk".format(args.output_dir), 'w')
    reco2dur_file = open("{}/reco2dur".format(args.output_dir), 'w')
    utt2numsamples_file = open("{}/utt2num_samples".format(args.output_dir), 'w')

    cnt, cnt_skip = 0, 0
    for utt_idx, utt in enumerate(uttlist):
        if utt not in utt2segs:
            print("{} not in segments, skipping it".format(utt))
            continue
        audio_path = utt2path[utt]
        segs = utt2segs[utt]
        print("Utt {}, {} segments".format(utt, len(segs)))

        try:
            if audio_path.endswith('.wav'):
                audio, sr = sf.read(audio_path)
            elif audio_path.endswith('|'):
                p = subprocess.Popen(audio_path[:-1], shell=True, stdout=subprocess.PIPE)
                audio, sr = sf.read(io.BytesIO(p.stdout.read()), dtype="float32")
            else:
                raise ValueError("Unsupported audio path format: {}".format(audio_path))
            assert sr == 16000, "Expected 16 kHz audio, got {} Hz: {}".format(sr, audio_path)
        except Exception as e:
            print("Error processing {}: {}".format(audio_path, e))
            continue

        if len(audio.shape) == 1:
            audio = audio[:, np.newaxis]

        for seg_idx, seg in enumerate(segs):
            try:
                segname, start_t, end_t = seg
                assert end_t > start_t
                cnt += 1
                if end_t - start_t < args.min_dur or end_t - start_t > args.max_dur:
                    print("Skipping {} duration {:.2f}".format(segname, end_t - start_t))
                    cnt_skip += 1
                    continue
                if seg2text[segname] == '':
                    print("Skipping {} empty text".format(segname))
                    cnt_skip += 1
                    continue
                start_sample, end_sample = int(start_t * sr), int(end_t * sr)
                assert end_sample <= audio.shape[0]
                sf.write("{}/wav/{}.wav".format(args.output_dir, segname), audio[start_sample:end_sample, :], samplerate=sr)

                wav_scp_file.write("{} {}/wav/{}.wav\n".format(segname, args.output_dir, segname))
                text_file.write("{} {}\n".format(segname, seg2text[segname]))
                utt2spk_file.write("{} {}\n".format(segname, segname))
                reco2dur_file.write("{} {:.2f}\n".format(segname, end_t - start_t))
                utt2numsamples_file.write("{} {}\n".format(segname, end_sample - start_sample))
            except Exception as e:
                segname, start_t, end_t = seg
                print("Error saving {}, start_t {:.2f}, end_t {:.2f}: {}".format(segname, start_t, end_t, e))
                cnt += 1
                cnt_skip += 1
                continue

    if cnt > 0:
        print("Skipping {} / {} = {:.2f}% segments".format(cnt_skip, cnt, 100.0 * cnt_skip / cnt))
    else:
        print("No segments processed.")

    wav_scp_file.close()
    text_file.close()
    utt2spk_file.close()
    reco2dur_file.close()
    utt2numsamples_file.close()
    return 0

if __name__ == '__main__':
    main()
