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
    utt2path = {}
    with open(fname, 'r') as fh:
        content = fh.readlines()
    for line in content:
        line = line.strip('\n')
        utt, path = line.split(None, 1)
        utt2path[utt] = path
    return utt2path

def get_utt2spk(fname):
    seg2spk = {}
    with open(fname, 'r') as fh:
        content = fh.readlines()
    for line in content:
        line = line.strip('\n')
        seg, spk = line.split()
        seg2spk[seg] = spk
    return seg2spk

def get_text(fname):
    seg2text = {}
    with open(fname, 'r') as fh:
        content = fh.readlines()
    for line in content:
        line = line.strip('\n')
        seg, text = line.split(None, 1)
        seg2text[seg] = text
    return seg2text

def get_segments(fname):
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
    if not os.path.exists(args.output_dir + '/wav'):
        os.makedirs(args.output_dir + '/wav')
    utt2path = get_wav_scp("{}/wav.scp".format(args.input_dir))
    seg2text = get_text("{}/text".format(args.input_dir))
    #seg2spk = get_utt2spk("{}/utt2spk".format(args.input_dir))
    utt2segs = get_segments("{}/segments".format(args.input_dir))
    for utt in utt2segs:
        if utt not in utt2path:
            print("{} not in wav.scp, skipping it".format(utt))
    #assert len(utt2path) == len(utt2segs)
    uttlist = list(utt2path.keys())
    uttlist.sort()

    wav_scp_file = open("{}/wav.scp".format(args.output_dir), 'w')
    text_file = open("{}/text".format(args.output_dir), 'w')
    utt2spk_file = open("{}/utt2spk".format(args.output_dir), 'w')
    reco2dur_file = open("{}/reco2dur".format(args.output_dir), 'w')
    utt2numsamples_file = open("{}/utt2num_samples".format(args.output_dir), 'w')
    
    cnt, cnt_skip = 0, 0
    for utt_idx, utt in enumerate(uttlist):
        audio_path = utt2path[utt]     
        segs = utt2segs[utt]
        print("Utt {}, {} segments".format(utt, len(segs)))

        try:
            chs = []
            if audio_path.endswith('.wav'):
                audio, sr = sf.read(audio_path)
            elif audio_path.endswith('|'):
                p = subprocess.Popen(audio_path[:-1], shell=True, stdout=subprocess.PIPE)
                audio, sr = sf.read(io.BytesIO(p.stdout.read()), dtype="float32")
            else:
                raise ValueError("Condition not defined.")
            chs.append(audio)
        except:
            print("Error processing {}, skipping it".format(audio_path))
            continue

        audio = np.stack(chs, axis=1)

        for seg_idx, seg in enumerate(segs):
            try:
                segname, start_t, end_t = seg
                assert end_t > start_t
                cnt += 1
                if end_t - start_t < args.min_dur or end_t - start_t > args.max_dur:
                    print("Skipping {} duration {:.2f}".format(segname, end_t - start_t))
                    cnt_skip += 1
                    continue
                start_sample, end_sample = int(start_t * 16000), int(end_t * 16000)
                assert end_sample <= audio.shape[0]
                sf.write("{}/wav/{}.wav".format(args.output_dir, segname), audio[start_sample:end_sample, :], samplerate=16000)

                # create wav.scp, reco2dur, segments, utt2spk files
                wav_scp_file.write("{} {}/wav/{}.wav\n".format(segname, args.output_dir, segname))
                text_file.write("{} {}\n".format(segname, seg2text[segname]))
                utt2spk_file.write("{} {}\n".format(segname, segname))
                reco2dur_file.write("{} {:.2f}\n".format(segname, end_t-start_t))
                utt2numsamples_file.write("{} {}\n".format(segname, end_sample-start_sample))
            except:
                segname, start_t, end_t = seg
                print("Error saving {}, start_t {:.2f}, end_t {:.2f}".format(segname, start_t, end_t)) 
                cnt += 1
                cnt_skip += 1
                continue
    print("Skipping {} / {} = {:.2f}% segments".format(cnt_skip, cnt, 100.0 * cnt_skip / cnt))

    wav_scp_file.close()
    text_file.close()
    utt2spk_file.close()
    reco2dur_file.close()
    utt2numsamples_file.close()
    return 0

if __name__ == '__main__':
    main()
