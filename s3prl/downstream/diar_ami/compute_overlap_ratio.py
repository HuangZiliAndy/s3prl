import os
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Compute speaker overlap')
parser.add_argument('reco2dur_file', type=str, help='reco2dur file')
parser.add_argument('rttm_file', type=str, help='rttm file')
parser.add_argument('utt2overlap_file', type=str, help='output utt2overlap file')
args = parser.parse_args()

def load_reco2dur(fname):
    reco2dur = {}
    with open(fname, 'r') as fh:
        content = fh.readlines()
    for line in content:
        line = line.strip('\n')
        line_split = line.split()
        reco, dur = line_split[0], float(line_split[1])
        reco2dur[reco] = dur
    return reco2dur

def load_rttm(fname):
    utt2segs = {}
    with open(fname, 'r') as fh:
        content = fh.readlines()
    for line in content:
        line = line.strip('\n')
        line_split = line.split()
        utt, start_t, duration, spk = line_split[1], float(line_split[3]), float(line_split[4]), line_split[7]
        end_t = start_t + duration
        start_t, end_t = round(start_t, 2), round(end_t, 2)
        if utt not in utt2segs:
            utt2segs[utt] = []
        utt2segs[utt].append([start_t, end_t, spk])
    return utt2segs

def get_spkmat(seg_list, duration):
    num_frames = int(duration * 100.0)
    spk_list = [seg[2] for seg in seg_list]
    spk_list = list(set(spk_list))
    spk_list.sort()
    spkmat = np.zeros((len(spk_list), num_frames))
    for seg in seg_list:
        start_t, end_t, spk = seg
        start_frame, end_frame = int(start_t * 100.0), int(end_t * 100.0)
        spkmat[spk_list.index(spk), start_frame:end_frame] = 1
    return spkmat

def main():
    reco2dur = load_reco2dur(args.reco2dur_file)
    utt2segs = load_rttm(args.rttm_file)
    assert len(reco2dur) == len(utt2segs)
    uttlist = list(reco2dur.keys())
    uttlist.sort()
    utt2overlap_file = open(args.utt2overlap_file, 'w')
    for utt in uttlist:
        spkmat = get_spkmat(utt2segs[utt], reco2dur[utt])
        activity = np.sum(spkmat, 0)
        speech = np.sum(activity > 0)
        overlap = np.sum(activity > 1)
        overlap_ratio = 100.0 * overlap / speech
        overlap_ratio1 = 100.0 * overlap / len(activity)
        utt2overlap_file.write("{} {:.2f}\n".format(utt, overlap_ratio))
    utt2overlap_file.close()
    return 0

if __name__ == '__main__':
    main()
