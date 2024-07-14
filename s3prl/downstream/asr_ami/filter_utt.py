import os
import sys
import argparse

parser = argparse.ArgumentParser(description='Filter out too short and too long utterances')
parser.add_argument('src_dir', type=str, help='Source directory')
parser.add_argument('tgt_dir', type=str, help='Target directory')
parser.add_argument('--min_dur', type=float, default=0.1, help='Minimum duration in second')
parser.add_argument('--max_dur', type=float, default=20.0, help='Maximum duration in second')
args = parser.parse_args()

def main():
    if not os.path.exists(args.tgt_dir):
        os.makedirs(args.tgt_dir)
    assert os.path.exists("{}/utt2num_samples".format(args.src_dir)) and os.path.exists("{}/text".format(args.src_dir)) and os.path.exists("{}/wav.scp".format(args.src_dir))
    with open("{}/utt2num_samples".format(args.src_dir), 'r') as fh:
        content_utt2num_samples = fh.readlines()
    with open("{}/text".format(args.src_dir), 'r') as fh:
        content_text = fh.readlines()
    with open("{}/wav.scp".format(args.src_dir), 'r') as fh:
        content_wav_scp = fh.readlines()
    assert len(content_utt2num_samples) == len(content_text) and len(content_text) == len(content_wav_scp)
    min_samples, max_samples = 16000 * args.min_dur, 16000 * args.max_dur
    
    utt2num_samples_file = open("{}/utt2num_samples".format(args.tgt_dir), 'w')
    text_file = open("{}/text".format(args.tgt_dir), 'w')
    wav_scp_file = open("{}/wav.scp".format(args.tgt_dir), 'w')
    cnt_skip = 0
    for i in range(len(content_utt2num_samples)):
        samples = int(content_utt2num_samples[i].strip('\n').split()[-1])
        if samples < min_samples or samples > max_samples:
            cnt_skip += 1
            continue
        else:
            utt2num_samples_file.write(content_utt2num_samples[i])
            text_file.write(content_text[i])
            wav_scp_file.write(content_wav_scp[i])
    utt2num_samples_file.close()
    text_file.close()
    wav_scp_file.close()
    print("Skipping {}/{} utterances".format(cnt_skip, len(content_utt2num_samples)))
    return 0

if __name__ == '__main__':
    main()
