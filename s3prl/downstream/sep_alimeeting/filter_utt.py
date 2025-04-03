import os
import sys
import argparse

parser = argparse.ArgumentParser(description='Filter out too short and too long utterances')
parser.add_argument('src_dir', type=str, help='Source directory')
parser.add_argument('tgt_dir', type=str, help='Target directory')
parser.add_argument('--min_dur', type=float, default=None, help='Min duration in second')
parser.add_argument('--max_dur', type=float, default=None, help='Max duration in second')
args = parser.parse_args()

def main():
    if not os.path.exists(args.tgt_dir):
        os.makedirs(args.tgt_dir)
    assert os.path.exists("{}/reco2dur".format(args.src_dir)) and os.path.exists("{}/wav.scp".format(args.src_dir))
    with open("{}/reco2dur".format(args.src_dir), 'r') as fh:
        content_reco2dur = fh.readlines()
    with open("{}/wav.scp".format(args.src_dir), 'r') as fh:
        content_wav_scp = fh.readlines()
    with open("{}/utt2spk".format(args.src_dir), 'r') as fh:
        content_utt2spk = fh.readlines()
    
    utt2spk_file = open("{}/utt2spk".format(args.tgt_dir), 'w')
    wav_scp_file = open("{}/wav.scp".format(args.tgt_dir), 'w')
    reco2dur_file = open("{}/reco2dur".format(args.tgt_dir), 'w')
    cnt_skip = 0
    total_dur, keep_dur = 0.0, 0.0
    for i in range(len(content_reco2dur)):
        duration = float((content_reco2dur[i].strip('\n')).split()[1])
        total_dur += duration
        if args.min_dur is not None and duration < args.min_dur:
            cnt_skip += 1
            continue
        if args.max_dur is not None and duration > args.max_dur:
            cnt_skip += 1
            continue
        utt2spk_file.write(content_utt2spk[i])
        wav_scp_file.write(content_wav_scp[i])
        reco2dur_file.write(content_reco2dur[i])
        keep_dur += duration
    print("Skipping {}/{} utterances".format(cnt_skip, len(content_reco2dur)))
    print("Keeping {:.2f}h/{:.2f}h = {:.2f}% duration".format(keep_dur / 3600.0, total_dur / 3600.0, 100 * keep_dur / total_dur))
    return 0

if __name__ == '__main__':
    main()
