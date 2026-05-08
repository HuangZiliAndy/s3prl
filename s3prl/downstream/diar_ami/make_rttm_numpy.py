#!/usr/bin/env python3

import argparse
import h5py
import numpy as np
import os
from scipy.signal import medfilt

parser = argparse.ArgumentParser(description="make rttm from decoded result")
parser.add_argument("score_dir")
parser.add_argument("out_rttm_file")
parser.add_argument("--threshold", default=0.5, type=float)
parser.add_argument("--frame_shift", default=320, type=int)
parser.add_argument("--subsampling", default=1, type=int)
parser.add_argument("--median", default=1, type=int)
parser.add_argument("--sampling_rate", default=16000, type=int)
args = parser.parse_args()

def main():
    numpy_files = ["{}/{}".format(args.score_dir, f) for f in os.listdir(args.score_dir) if f.endswith('.npy')]
    numpy_files.sort()
    print("Create RTTM for {} utterances".format(len(numpy_files)))
    with open(args.out_rttm_file, "w") as wf:
        for filepath in numpy_files:
            session, _ = os.path.splitext(os.path.basename(filepath))
            data = np.load(filepath)
            a = np.where(data[:] > args.threshold, 1, 0)
            if args.median > 1:
                a = medfilt(a, (args.median, 1))
            factor = args.frame_shift * args.subsampling / args.sampling_rate
            for spkid, frames in enumerate(a.T):
                frames = np.pad(frames, (1, 1), "constant")
                (changes,) = np.where(np.diff(frames, axis=0) != 0)
                fmt = "SPEAKER {:s} 1 {:7.2f} {:7.2f} <NA> <NA> {:s} <NA>"
                for s, e in zip(changes[::2], changes[1::2]):
                    print(
                        fmt.format(
                            session,
                            s * factor,
                            (e - s) * factor,
                            session + "_" + str(spkid),
                        ),
                        file=wf,
                    )
    return 0

if __name__ == '__main__':
    main()
