import os
import io
import argparse
import shutil
import soundfile as sf
import numpy as np
import subprocess 

parser = argparse.ArgumentParser(description='Create diarization dataset')
parser.add_argument('input_dir', type=str, help='Input directory')
parser.add_argument('output_dir', type=str, help='Output directory')
parser.add_argument('--chunk_size', type=float, default=10.0, help='chunk size')
parser.add_argument('--stride_size', type=float, default=10.0, help='stride size')
parser.add_argument('--normalize', type=int, default=0, help='whether to normalize audio')

args = parser.parse_args()

def get_wav_scp(fname):
    utt2path = {}
    with open(fname, 'r') as fh:
        content = fh.readlines()
    for line in content:
        line = line.strip('\n')
        utt, path = line.split(None, 1)
        utt2path[utt] = path
    return utt2path

def get_rttm_scp(fname):
    utt2seg = {}
    with open(fname, 'r') as fh:
        content = fh.readlines()
    for line in content:
        line = line.strip('\n')
        line_split = line.split()
        utt, rttm_file = line_split[0], line_split[1]
        with open(rttm_file, 'r') as fh:
            rttm_content = fh.readlines()
        utt2seg[utt] = []
        for line in rttm_content:
            line = line.strip('\n')
            line_split = line.split()
            start_t, dur = float(line_split[3]), float(line_split[4])
            end_t = start_t + dur
            start_t, end_t = round(start_t, 2), round(end_t, 2)
            spk = line_split[7]
            utt2seg[utt].append([start_t, end_t, spk])
    return utt2seg

def main():
    if not os.path.exists(args.output_dir + '/wav'):
        os.makedirs(args.output_dir + '/wav')
    if not os.path.exists(args.output_dir + '/rttm'):
        os.makedirs(args.output_dir + '/rttm')
    utt2path = get_wav_scp("{}/wav.scp".format(args.input_dir))
    utt2seg = get_rttm_scp("{}/rttm.scp".format(args.input_dir))

    uttlist = list(utt2path.keys())
    uttlist.sort()

    wav_scp_file = open("{}/wav.scp".format(args.output_dir), 'w')
    rttm_scp_file = open("{}/rttm.scp".format(args.output_dir), 'w')
    reco2dur_file = open("{}/reco2dur".format(args.output_dir), 'w')
    utt2numspks_file = open("{}/utt2numspks".format(args.output_dir), 'w')

    numspk_per_seg = {}
    
    if args.normalize:
        print("Normalizing audio")
    else:
        print("Not normalizing audio")

    for utt_idx, utt in enumerate(uttlist):
        print("{}/{}".format(utt_idx+1, len(uttlist)))
        audio_path = utt2path[utt]     
        segs = utt2seg[utt]
        print("Utt {}, {} segments".format(utt, len(segs)))

        if audio_path.endswith('.wav'):
            audio, sr = sf.read(audio_path)
        elif audio_path.endswith('|'):
            p = subprocess.Popen(audio_path[:-1], shell=True, stdout=subprocess.PIPE)
            audio, sr = sf.read(io.BytesIO(p.stdout.read()), dtype="float32")
        else:
            raise ValueError("Condition not defined.")
        if len(audio.shape) == 1:
            audio = np.expand_dims(audio, axis=1)

        if args.normalize:
            audio = audio / (np.max(np.abs(audio)) + 1e-10)

        recodur = audio.shape[0] / 16000.0
        start_t, end_t = 0, 0
        while start_t <= recodur - args.chunk_size:
            end_t = start_t + args.chunk_size
            start_sample, end_sample = int(start_t * 16000), int(end_t * 16000)
            segname = "{}_{:06d}_{:06d}".format(utt, int(round(start_t * 100.0)), int(round(end_t * 100.0)))

            # skip invalid segments
            if np.max(np.abs(audio[start_sample:end_sample, :])) == 0.0:
                start_t += args.stride_size
                print("Invalid segment {}, skipping it".format(segname))
                continue

            segs_between = [seg for seg in segs if seg[0] < end_t and seg[1] > start_t] 
            segs_between = [[max(0, seg[0]-start_t), min(seg[1]-start_t, args.chunk_size), seg[2]] for seg in segs_between]
            num_spks = len(set([seg[2] for seg in segs_between]))
            if num_spks not in numspk_per_seg:
                numspk_per_seg[num_spks] = 0
            numspk_per_seg[num_spks] += 1

            # create wav file            
            sf.write("{}/wav/{}.wav".format(args.output_dir, segname), audio[start_sample:end_sample, :], samplerate=16000)

            # create RTTM file
            with open("{}/rttm/{}.rttm".format(args.output_dir, segname), 'w') as fh:
                for seg in segs_between:
                    fh.write("SPEAKER {0} {1} {2:7.2f} {3:7.2f} <NA> <NA> {4} <NA>\n".format(segname, 1, seg[0], seg[1] - seg[0], seg[2]))

            # create wav.scp, reco2dur, segments, utt2spk files
            wav_scp_file.write("{} {}/wav/{}.wav\n".format(segname, args.output_dir, segname))
            rttm_scp_file.write("{} {}/rttm/{}.rttm\n".format(segname, args.output_dir, segname))
            reco2dur_file.write("{} {:.2f}\n".format(segname, end_t-start_t))
            utt2numspks_file.write("{} {}\n".format(segname, num_spks))
             
            start_t += args.stride_size

    total_segs = np.sum(list(numspk_per_seg.values()))
    print("Total {} segments".format(total_segs))
    for k in numspk_per_seg.keys():
        print("{} speakers {} segments ({:.2f}%)".format(k, numspk_per_seg[k], 100.0 * numspk_per_seg[k] / total_segs))

    wav_scp_file.close()
    rttm_scp_file.close()
    reco2dur_file.close()
    utt2numspks_file.close()
    return 0

if __name__ == '__main__':
    main()
