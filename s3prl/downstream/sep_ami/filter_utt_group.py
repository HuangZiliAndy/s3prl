import os
import sys
import argparse

parser = argparse.ArgumentParser(description='Filter utterances according to number of speakers')
parser.add_argument('src_dir', type=str, help='Source directory')
parser.add_argument('tgt_dir', type=str, help='Target directory')
parser.add_argument('--max_num_spks', type=int, default=1, help='Maximum number of speakers')
parser.add_argument('--min_num_spks', type=int, default=1, help='Minimum number of speakers')
args = parser.parse_args()

def get_utt2numspks(fname):
    utt2numspks = {}
    with open(fname, 'r') as fh:
        content = fh.readlines()
    for line in content:
        line = line.strip('\n')
        utt, numspks = line.split()
        utt2numspks[utt] = int(numspks)
    return utt2numspks

def main():
    if not os.path.exists(args.tgt_dir):
        os.makedirs(args.tgt_dir)
    fname_list = ["reco2dur", "rttm", "text", "utt2numspks", "utt2overlap", "utt2spk", "wav.scp"]
    for f in fname_list:
        assert os.path.exists("{}/{}".format(args.src_dir, f))

    utt2numspks = get_utt2numspks("{}/utt2numspks".format(args.src_dir))

    field_list = [0, 1, 0, 0, 0, 0, 0]
    for fname, field in zip(fname_list, field_list):
        with open("{}/{}".format(args.src_dir, fname), 'r') as fh:
            content = fh.readlines()
        output_file = open("{}/{}".format(args.tgt_dir, fname), 'w')
        for line in content:
            line = line.strip('\n')
            line_split = line.split()
            if line_split[field] in utt2numspks and utt2numspks[line_split[field]] <= args.max_num_spks and utt2numspks[line_split[field]] >= args.min_num_spks:
                output_file.write(line + '\n')
        output_file.close()
    if args.max_num_spks == args.min_num_spks:
        with open("{}/num_srcs".format(args.tgt_dir), 'w') as fh:
            fh.write("{}\n".format(args.max_num_spks))
    return 0

if __name__ == '__main__':
    main()
