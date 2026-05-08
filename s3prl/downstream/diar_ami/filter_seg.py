import os
import argparse

parser = argparse.ArgumentParser(description='Filter dataset based on the number of speakers')
parser.add_argument('input_dir', type=str, help='Input directory')
parser.add_argument('output_dir', type=str, help='Output directory')
parser.add_argument('--max_spks', type=int, default=4, help='Maximum number of speakers')
parser.add_argument('--min_spks', type=int, default=0, help='Minimum number of speakers')
args = parser.parse_args()

def load_mapping(fname):
    utt2info = {}
    with open(fname, 'r') as fh:
        content = fh.readlines()
    for line in content:
        line = line.strip('\n')
        line_split = line.split(None, 1)
        utt2info[line_split[0]] = line_split[1]
    return utt2info

def main():
    for f in ["wav.scp", "rttm.scp", "reco2dur", "utt2numspks"]:
        assert os.path.exists("{}/{}".format(args.input_dir, f))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    utt2path = load_mapping("{}/wav.scp".format(args.input_dir))
    utt2rttm = load_mapping("{}/rttm.scp".format(args.input_dir))
    utt2numspks = load_mapping("{}/utt2numspks".format(args.input_dir))
    reco2dur = load_mapping("{}/reco2dur".format(args.input_dir))
    uttlist = list(utt2path.keys())
    uttlist.sort()
    wav_scp_file = open("{}/wav.scp".format(args.output_dir), 'w')
    rttm_scp_file = open("{}/rttm.scp".format(args.output_dir), 'w')
    reco2dur_file = open("{}/reco2dur".format(args.output_dir), 'w')
    utt2numspks_file = open("{}/utt2numspks".format(args.output_dir), 'w')
    for utt in uttlist:
        numspks = int(utt2numspks[utt])
        if numspks >= args.min_spks and numspks <= args.max_spks:
            wav_scp_file.write("{} {}\n".format(utt, utt2path[utt]))
            rttm_scp_file.write("{} {}\n".format(utt, utt2rttm[utt]))
            reco2dur_file.write("{} {}\n".format(utt, reco2dur[utt]))
            utt2numspks_file.write("{} {}\n".format(utt, utt2numspks[utt]))
    wav_scp_file.close()
    rttm_scp_file.close()
    reco2dur_file.close()
    utt2numspks_file.close()
    return 0

if __name__ == '__main__':
    main()
