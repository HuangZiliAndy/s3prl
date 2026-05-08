import os
import numpy as np
import subprocess
import argparse

parser = argparse.ArgumentParser(description='Analyze diarization results')
parser.add_argument('test_dir', type=str, help='Test directory')
parser.add_argument('hyp_rttm', type=str, help='Predict rttm file')
args = parser.parse_args()

def get_utt2DER(test_dir, hyp_rttm):
    utt2DER, utt2miss, utt2fa, utt2cf, utt2scored = {}, {}, {}, {}, {}
    for f in ["ref_rttm", "uem", "utt2overlap"]:
        assert os.path.exists("{}/{}".format(test_dir, f))
    cmd = "./downstream/diar_ami/md-eval.pl -r {}/ref_rttm -s {} -u {}/uem -a f".format(test_dir, hyp_rttm, test_dir)
    (status, output) = subprocess.getstatusoutput(cmd)
    assert status == 0
    output_lines = output.split('\n')
    scored, miss, fa, cf = 0, 0, 0, 0 
    for line in output_lines:
        if line.startswith("SCORED SPEAKER TIME"):
            scored = float(line.split()[4])
        elif line.startswith("MISSED SPEAKER TIME"):
            miss = float(line.split()[4])
        elif line.startswith("FALARM SPEAKER TIME"):
            fa = float(line.split()[4])
        elif line.startswith(" SPEAKER ERROR TIME"):
            cf = float(line.split()[4])
        elif line.startswith(" OVERALL SPEAKER DIARIZATION ERROR"):
            line_split = line.split()
            DER = float(line_split[5])
            assert np.abs((miss + fa + cf) / scored * 100 - DER) <= 0.01
            if line_split[-1] == "`(ALL)":
                continue
            else:
                uttname = (line_split[-1].split('=')[1])[:-1]
            utt2DER[uttname] = DER
            utt2miss[uttname], utt2fa[uttname], utt2cf[uttname], utt2scored[uttname] = miss, fa, cf, scored
    cmd = "./downstream/diar_ami/md-eval.pl -r {}/ref_rttm -s {} -u {}/uem".format(test_dir, hyp_rttm, test_dir)
    (status, output) = subprocess.getstatusoutput(cmd)
    assert status == 0
    output_lines = output.split('\n')
    scored_all, miss_all, fa_all, cf_all = 0, 0, 0, 0 
    for line in output_lines:
        if line.startswith("SCORED SPEAKER TIME"):
            scored_all = float(line.split()[4])
        elif line.startswith("MISSED SPEAKER TIME"):
            miss_all = float(line.split()[4])
        elif line.startswith("FALARM SPEAKER TIME"):
            fa_all = float(line.split()[4])
        elif line.startswith(" SPEAKER ERROR TIME"):
            cf_all = float(line.split()[4])
        elif line.startswith(" OVERALL SPEAKER DIARIZATION ERROR"):
            line_split = line.split()
            DER_all = float(line_split[5])
            assert np.abs((miss_all + fa_all + cf_all) / scored_all * 100 - DER_all) <= 0.01
    cmd = "./downstream/diar_ami/md-eval.pl -r {}/ref_rttm -s {} -u {}/uem -1".format(test_dir, hyp_rttm, test_dir)
    (status, output) = subprocess.getstatusoutput(cmd)
    assert status == 0
    output_lines = output.split('\n')
    scored_nooverlap, miss_nooverlap, fa_nooverlap, cf_nooverlap = 0, 0, 0, 0 
    for line in output_lines:
        if line.startswith("SCORED SPEAKER TIME"):
            scored_nooverlap = float(line.split()[4])
        elif line.startswith("MISSED SPEAKER TIME"):
            miss_nooverlap = float(line.split()[4])
        elif line.startswith("FALARM SPEAKER TIME"):
            fa_nooverlap = float(line.split()[4])
        elif line.startswith(" SPEAKER ERROR TIME"):
            cf_nooverlap = float(line.split()[4])
        elif line.startswith(" OVERALL SPEAKER DIARIZATION ERROR"):
            line_split = line.split()
            DER_nooverlap = float(line_split[5])
            assert np.abs((miss_nooverlap + fa_nooverlap + cf_nooverlap) / scored_nooverlap * 100 - DER_nooverlap) <= 0.01
    DER_overlap = ((miss_all + fa_all + cf_all) - (miss_nooverlap + fa_nooverlap + cf_nooverlap)) / (scored_all - scored_nooverlap) * 100.0
    print("DER {:.2f}%, DER no overlap {:.2f}%, DER overlap {:.2f}%".format(DER_all, DER_nooverlap, DER_overlap))
    return utt2DER, utt2miss, utt2fa, utt2cf, utt2scored

def get_utt2overlap(fname):
    utt2overlap = {}
    with open(fname, 'r') as fh:
        content = fh.readlines()
    for line in content:
        line = line.strip('\n')
        line_split = line.split()
        utt2overlap[line_split[0]] = float(line_split[1])
    return utt2overlap

def main():
    utt2overlap = get_utt2overlap("{}/utt2overlap".format(args.test_dir))
    utt2DER, utt2miss, utt2fa, utt2cf, utt2scored = get_utt2DER(args.test_dir, args.hyp_rttm)
    assert len(utt2overlap) == len(utt2DER)
    overlap2error, overlap2scored = {}, {}
    uttlist = list(utt2overlap.keys())
    uttlist.sort()
    max_idx = -1
    for utt in uttlist:
        overlap = utt2overlap[utt]
        idx = int(np.floor(overlap / 5.0))
        if idx > max_idx:
            max_idx = idx
        if idx not in overlap2error:
            overlap2error[idx] = []
            overlap2scored[idx] = []
        overlap2error[idx].append(utt2miss[utt] + utt2fa[utt] + utt2cf[utt])
        overlap2scored[idx].append(utt2scored[utt])
    for i in range(0, max_idx + 1):
        print("Overlap ratio between {:.2f} to {:.2f}%".format(i * 5, (i+1) * 5))
        if i not in overlap2error:
            print("0 utts")
        else:
            print("{} utts, DER {:.2f}%".format(len(overlap2error[i]), 100.0 * np.sum(overlap2error[i]) / np.sum(overlap2scored[i])))
    return 0

if __name__ == '__main__':
    main()
