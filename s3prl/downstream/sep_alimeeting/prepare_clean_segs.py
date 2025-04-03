import os
import io
import argparse
import shutil
import soundfile as sf
import numpy as np
import subprocess 
import matplotlib.pyplot as plt
import textgrid

parser = argparse.ArgumentParser(description='Extract clean segments')
parser.add_argument('sdm1_dir', type=str, help='SDM1 directory')
parser.add_argument('output_dir', type=str, help='Output directory')
parser.add_argument('near_audio_dir', type=str, help='audio for near field speech')
parser.add_argument('far_text_dir', type=str, help='text for far field speech')
parser.add_argument('--overlap_thres', type=float, default=5.0, help='keep the segments whose overlap ratio is smaller than overlap_thres')
parser.add_argument('--min_dur', type=float, default=0.1, help='segment length should be longer than min_dur')
args = parser.parse_args()

def get_meetspk2channel(far_text_dir, near_audio_dir):
    meetspk2audiopath, meet2spks = {}, {}
    textgrid_files = list(os.listdir(far_text_dir))
    textgrid_files.sort()
    for textgrid_f in textgrid_files:
        tg = textgrid.TextGrid.fromFile('{}/{}'.format(far_text_dir, textgrid_f))
        meet = textgrid_f.rstrip('.TextGrid')
        spk_list = []
        for tier in tg.tiers:
            spk = tier.name
            spk = '_'.join([spk.split('_')[-2], spk.split('_')[-1]])
            audio_path = "{}/{}_{}.wav".format(near_audio_dir, meet, spk)
            assert os.path.exists(audio_path)
            meetspk2audiopath["{}_{}".format(meet, spk)] = audio_path
            spk_list.append(spk)
        meet2spks[meet] = spk_list
    return meetspk2audiopath, meet2spks

def get_wav_scp(fname):
    utt2path = {}
    with open(fname, 'r') as fh:
        content = fh.readlines()
    for line in content:
        line = line.strip('\n')
        utt, path = line.split(None, 1)
        utt2path[utt] = path
    return utt2path

def get_reco2dur(fname):
    reco2dur = {}
    with open(fname, 'r') as fh:
        content = fh.readlines()
    for line in content:
        line = line.strip('\n')
        utt, dur = line.split(None, 1)
        reco2dur[utt] = float(dur)
    return reco2dur

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

def get_clean_segments(duration, segs, spk_list, overlap_thres):
    total_frames = round(duration * 100.0)
    spk_matrix = np.zeros((total_frames, len(spk_list)))
    spkmap = {(spk.split('_')[-1]).lstrip('SPK'): spk for spk in spk_list}
    for seg in segs:
        start_t, end_t, spk = seg
        spk = spkmap[spk]
        start_frame, end_frame = round(start_t * 100.0), round(end_t * 100.0)
        spk_matrix[start_frame:end_frame, spk_list.index(spk)] = 1
    clean_segs = []
    for i, seg in enumerate(segs):
        start_t, end_t, spk = seg
        start_frame, end_frame = round(start_t * 100.0), round(end_t * 100.0)
        if end_frame > total_frames:
            continue
        activity_matrix = np.sum(spk_matrix[start_frame:end_frame, :], 1)
        overlap = np.sum(activity_matrix > 1)
        total = np.sum(activity_matrix >= 1)
        assert end_frame - start_frame == total 
        overlap_ratio = 100.0 * overlap / total
        if overlap_ratio <= overlap_thres:
            clean_segs.append(seg)
    return clean_segs 

def main():
    if not os.path.exists("{}/wav".format(args.output_dir)):
        os.makedirs("{}/wav".format(args.output_dir))
    wav_scp_file = open("{}/wav.scp".format(args.output_dir), 'w')
    utt2spk_file = open("{}/utt2spk".format(args.output_dir), 'w')
    reco2dur_file = open("{}/reco2dur".format(args.output_dir), 'w')

    meetspk2audiopath, meet2spks = get_meetspk2channel(args.far_text_dir, args.near_audio_dir)

    utt2path = get_wav_scp("{}/wav.scp".format(args.sdm1_dir))
    utt2seg = get_rttm_scp("{}/rttm.scp".format(args.sdm1_dir))
    reco2dur = get_reco2dur("{}/reco2dur".format(args.sdm1_dir))

    uttlist = list(utt2path.keys())
    uttlist.sort()

    numspk_per_seg = {}
    clean_segs = []

    for utt_idx, utt in enumerate(uttlist):
        print("{}/{}".format(utt_idx+1, len(uttlist)))
        audio_path = utt2path[utt]     
        segs = utt2seg[utt]
        recodur = reco2dur[utt]
        spklist = meet2spks[utt]
        print("Utt {} {:.2f} seconds, {} speakers, {} segments".format(utt, recodur, len(spklist), len(segs)))

        try:
            meetspk2audio = {}
            for spk in spklist:
                audio_path_spk = meetspk2audiopath["{}_{}".format(utt, spk)]
                if audio_path_spk.endswith('.wav'):
                    audio, sr = sf.read(audio_path_spk)
                elif audio_path_spk.endswith('|'):
                    p = subprocess.Popen(audio_path_spk[:-1], shell=True, stdout=subprocess.PIPE)
                    audio, sr = sf.read(io.BytesIO(p.stdout.read()), dtype="float32")
                else:
                    raise ValueError("Condition not defined.")
                assert len(audio.shape) == 1
                ihm_spk_audio = np.expand_dims(audio, axis=1)
                meetspk2audio["{}_{}".format(utt, spk)] = ihm_spk_audio
        except:
            print("Error processing {}, skipping it".format(audio_path))
            continue

        clean_segs_utt = get_clean_segments(recodur, segs, spklist, args.overlap_thres)

        clean_segs_utt = [seg for seg in clean_segs_utt if seg[1] - seg[0] >= args.min_dur]

        spkmap = {(spk.split('_')[-1]).lstrip('SPK'): spk for spk in spklist}
        # write the segment to disk
        for seg in clean_segs_utt:
            start_t, end_t, spk = seg
            spk = spkmap[spk]
            start_t, end_t = round(start_t, 2), round(end_t, 2)
            start_sample, end_sample = int(round(start_t * 16000.0)), int(round(end_t * 16000.0))
            segname = "{}_{}_{:07d}_{:07d}".format(utt, spk, int(round(start_t * 100)), int(round(end_t * 100)))
            sf.write('{}/wav/{}.wav'.format(args.output_dir, segname), meetspk2audio["{}_{}".format(utt, spk)][start_sample:end_sample], 16000)
            wav_scp_file.write("{} {}/wav/{}.wav\n".format(segname, args.output_dir, segname))
            utt2spk_file.write("{} {}\n".format(segname, spk))
            reco2dur_file.write("{} {:.2f}\n".format(segname, end_t - start_t))

        clean_segs += clean_segs_utt

    seg_len_list = [seg[1] - seg[0] for seg in clean_segs]
    print("Total duration {:.2f} hours".format(np.sum(seg_len_list) / 3600.0))

    for duration in [2, 3, 4]:
        print("Segments longer than {}s {} hours".format(duration, np.sum([seg_len for seg_len in seg_len_list if seg_len >= duration]) / 3600.0))

    plt.hist(seg_len_list, bins=50, edgecolor='black')
    plt.savefig('{}/segment_len.jpg'.format(args.output_dir), format='jpg')

    wav_scp_file.close()
    utt2spk_file.close()
    reco2dur_file.close()
    return 0

if __name__ == '__main__':
    main()
