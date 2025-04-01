import os
import time
import argparse
import subprocess
import soundfile as sf
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser(description='Beamformit')
parser.add_argument('wav_scp_file', type=str, help='wav scp file')
parser.add_argument('output_dir', type=str, help='output directory')
parser.add_argument('--config_file', type=str, help='config file')
parser.add_argument('--split', type=int, default=1, help='split number')
parser.add_argument('--rank', type=int, default=0, help='rank number')
parser.add_argument('--num_jobs', type=int, default=8, help='number of jobs')
parser.add_argument('--channels', type=str, help='')
args = parser.parse_args()

def batch_process(lines, num_jobs):
    results = Parallel(n_jobs=num_jobs)(
        delayed(beamform)(line)
        for line in tqdm(lines)
    )
    return results

def pad_speech(speech_path, nsamples):
    speech, sr = sf.read(speech_path)
    if len(speech) < nsamples:
        padded_speech = np.zeros((nsamples, ))
        padded_speech[:len(speech)] = speech
    else:
        padded_speech = speech
    sf.write(speech_path, padded_speech, sr)
    return 0

def trim_speech(speech_path, nsamples):
    speech, sr = sf.read(speech_path)
    sf.write(speech_path, speech[:nsamples], sr)
    return 0

def beamform(line):
    line = line.strip('\n')
    line_split = line.split()
    uttname, wav_path = line_split[0], line_split[1]
    assert os.path.exists(wav_path)
    channel_str = "{}".format(uttname)
    num_samples = (sf.info(wav_path)).frames

    if args.channels != '':
        nchans = [int(c) for c in args.channels.split(',')]
    else:
        nchans = list(range(sf.info(wav_path).channels))

    min_samples = 16000
    # temporary store selected channels
    for c in nchans:
        cmd = "sox {} {}/tmp_{}_{}/{}_{}.wav remix {}".format(wav_path, args.output_dir, args.split, args.rank, uttname, c, c+1)
        status, output = subprocess.getstatusoutput(cmd)
        assert status == 0
        if num_samples < min_samples:
            pad_speech("{}/tmp_{}_{}/{}_{}.wav".format(args.output_dir, args.split, args.rank, uttname, c), min_samples)
        channel_str += " " + "{}_{}.wav".format(uttname, c) 
    channel_str += '\n'
    channel_file = "{}/tmp_{}_{}/channels_{}".format(args.output_dir, args.split, args.rank, uttname)
    with open(channel_file, 'w') as fh:
        fh.write(channel_str)
    cmd = "BeamformIt -s {} -c {} --config_file {} --source_dir {}/tmp_{}_{} --result_dir {}/wav".format(uttname, channel_file, args.config_file, args.output_dir, args.split, args.rank, args.output_dir)
    status, output = subprocess.getstatusoutput(cmd)
    assert status == 0
    
    if num_samples < min_samples:
        trim_speech("{}/wav/{}.wav".format(args.output_dir, uttname), num_samples)

    # remove extra files
    for c in nchans:
        os.remove("{}/tmp_{}_{}/{}_{}.wav".format(args.output_dir, args.split, args.rank, uttname, c))
    for ext in ["del", "del2", "info", "weat"]:
        os.remove("{}/wav/{}.{}".format(args.output_dir, uttname, ext))
    return 0

def main():
    print(args)
    if not os.path.exists(args.output_dir + '/tmp_{}_{}'.format(args.split, args.rank)):
        os.makedirs(args.output_dir + '/tmp_{}_{}'.format(args.split, args.rank))
    if not os.path.exists(args.output_dir + '/wav'):
        os.makedirs(args.output_dir + '/wav')
    with open(args.wav_scp_file, 'r') as fh:
        content = fh.readlines()
    cnt = 0
    start_time = time.time()
    content_rank = [line for i, line in enumerate(content) if i % args.split == args.rank]
    print("Rank {}, process {} / {} utts".format(args.rank, len(content_rank), len(content)))

    batch_process(content_rank, num_jobs=args.num_jobs)
    #for line in tqdm(content_rank):
    #    beamform(line)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.6f} seconds")
    return 0

if __name__ == '__main__':
    main()
