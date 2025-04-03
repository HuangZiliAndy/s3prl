import os
import io
import numpy as np
import soundfile as sf
import subprocess
import argparse

parser = argparse.ArgumentParser(description='Create utterance group')
parser.add_argument('input_dir', type=str, help='input directory')
parser.add_argument('output_dir', type=str, help='output directory')
parser.add_argument('--min_dur', type=float, default=0.1, help='min duration')
parser.add_argument('--max_dur', type=float, default=10000.0, help='min duration')
parser.add_argument('--normalize', type=int, default=0, help='whether to normalize audio')
args = parser.parse_args()

def get_utt2path(wav_scp_file):
    utt2path = {}
    with open(wav_scp_file, 'r') as fh:
        content = fh.readlines()
    for line in content:
        line = line.strip('\n')
        line_split = line.split(None, 1)
        utt, path = line_split[0], line_split[1]
        utt2path[utt] = path
    return utt2path

def get_meet2spks(segments_file):
    meet2spks = {}
    with open(segments_file, 'r') as fh:
        content = fh.readlines()
    for line in content:
        line = line.strip('\n')
        line_split = line.split()
        segname, meet = line_split[0], line_split[1]
        spk = segname.split('_')[-3]
        if meet not in meet2spks:
            meet2spks[meet] = []
        if spk not in meet2spks[meet]:
            meet2spks[meet].append(spk)
    for meet in meet2spks.keys():
        meet2spks[meet].sort()
    return meet2spks

def get_utt2seg(segments_file, text_file):
    utt2seg = {}
    with open(segments_file, 'r') as fh:
        content_segments = fh.readlines()
    with open(text_file, 'r') as fh:
        content_text = fh.readlines()
    assert len(content_segments) == len(content_text)
    for i in range(len(content_segments)):
        line = content_segments[i].strip('\n')
        line_split = line.split()
        seg, utt, start_t, end_t = line_split[0], line_split[1], round(float(line_split[2]), 2), round(float(line_split[3]), 2)
        spk = seg.split('_')[-3]
        text_line = content_text[i].strip('\n')
        text_line_split = text_line.split(None, 1)
        if utt not in utt2seg:
            utt2seg[utt] = []
        utt2seg[utt].append([start_t, end_t, spk, text_line_split[1]])
    return utt2seg

def merge_seg(seg_list):
    seg_list_sort = sorted(seg_list, key=lambda x: x[0])
    #seg_list_sort = seg_list_sort[:20]
    uttgroup_list = []
    start_t, end_t, spk_list, text_list, timebound_list = None, None, [], [], []
    for i in range(len(seg_list_sort)):
        if start_t is None and end_t is None:
            start_t = seg_list_sort[i][0]
            end_t = seg_list_sort[i][1]
            spk_list.append(seg_list_sort[i][2])
            text_list.append(seg_list_sort[i][3])
            timebound_list.append([seg_list_sort[i][0], seg_list_sort[i][1]])
        elif seg_list_sort[i][0] >= end_t:
            uttgroup_list.append([start_t, end_t, spk_list, text_list, timebound_list])
            start_t = seg_list_sort[i][0]
            end_t = seg_list_sort[i][1]
            spk_list = [seg_list_sort[i][2]]
            text_list = [seg_list_sort[i][3]]
            timebound_list = [[seg_list_sort[i][0], seg_list_sort[i][1]]]
        else:
            end_t = max(end_t, seg_list_sort[i][1])
            spk_list.append(seg_list_sort[i][2])
            text_list.append(seg_list_sort[i][3])
            timebound_list.append([seg_list_sort[i][0], seg_list_sort[i][1]])
    uttgroup_list.append([start_t, end_t, spk_list, text_list, timebound_list])
    return uttgroup_list

def compute_overlap_ratio(spk_list, timebound_list, start_t):
    new_timebound_list = [[timebound[0] - start_t, timebound[1] - start_t] for timebound in timebound_list]
    max_time = np.max(np.array(new_timebound_list))
    max_frame = int(max_time * 100.0)
    spk_set = list(set(spk_list))
    assert len(spk_set) > 0
    if len(spk_set) == 1:
        return 0
    else:
        spk_mat = np.zeros((max_frame, len(spk_set)))
        for spk, timebound in zip(spk_list, new_timebound_list):
            start_frame, end_frame = round(timebound[0] * 100.0), round(timebound[1] * 100.0)
            spk_mat[start_frame:end_frame, spk_set.index(spk)] = 1
        spk_activity = np.sum(spk_mat, 1)
        overlap_ratio = np.sum(spk_activity >= 2) / np.sum(spk_activity >= 1) * 100.0
        return overlap_ratio

def main():
    meet2spks = get_meet2spks("{}/segments".format(args.input_dir))
    utt2seg = get_utt2seg("{}/segments".format(args.input_dir), "{}/text".format(args.input_dir))
    utt2path = get_utt2path("{}/wav.scp".format(args.input_dir))
    uttlist = list(utt2seg.keys())
    uttlist.sort()

    if not os.path.exists(args.output_dir + '/wav'):
        os.makedirs(args.output_dir + '/wav')

    wav_scp_file = open("{}/wav.scp".format(args.output_dir), 'w')
    text_file = open("{}/text".format(args.output_dir), 'w')
    utt2spk_file = open("{}/utt2spk".format(args.output_dir), 'w')
    rttm_file = open("{}/rttm".format(args.output_dir), 'w')
    reco2dur_file = open("{}/reco2dur".format(args.output_dir), 'w')
    utt2numspks_file = open("{}/utt2numspks".format(args.output_dir), 'w')
    utt2overlap_file = open("{}/utt2overlap".format(args.output_dir), 'w')

    cnt_short, cnt_long, cnt = 0, 0, 0
    for utt in uttlist:
        audio_path = utt2path[utt]
        if audio_path.endswith('.wav'):
            audio, sr = sf.read(audio_path)
        elif audio_path.endswith('|'):
            p = subprocess.Popen(audio_path[:-1], shell=True, stdout=subprocess.PIPE)
            audio, sr = sf.read(io.BytesIO(p.stdout.read()), dtype="float32")
        else:
            raise ValueError("Condition not defined.")

        if args.normalize:
            print("Normalizing audio")
            audio = audio / np.max(np.abs(audio))
        else:
            print("Not normalizing audio")

        uttgroup_list = merge_seg(utt2seg[utt]) 
        all_spks = meet2spks[utt]
        print("Utt {}, {} segments, {} utt groups, {} spks".format(utt, len(utt2seg[utt]), len(uttgroup_list), len(all_spks)))

        for uttgroup in uttgroup_list:
            cnt += 1
            start_t, end_t, spk_list, text_list, timebound_list = uttgroup[0], uttgroup[1], uttgroup[2], uttgroup[3], uttgroup[4]
            assert len(spk_list) == len(text_list) == len(timebound_list)
            if end_t - start_t < args.min_dur:
                cnt_short += 1
                continue
            if end_t - start_t > args.max_dur:
                cnt_long += 1
                continue
        
            # write wav file
            wavname = "{}-{:07d}-{:07d}".format(utt, int(start_t * 100.0), int(end_t * 100.0))
            start_sample, end_sample = int(start_t * 16000.0), int(end_t * 16000.0)
            sf.write("{}/wav/{}.wav".format(args.output_dir, wavname), audio[start_sample:end_sample], 16000)
            wav_scp_file.write("{} {}/wav/{}.wav\n".format(wavname, args.output_dir, wavname))
            
            # write metadata files
            all_texts = [[] for i in range(len(all_spks))]
            for i in range(len(spk_list)):
                spk, text = spk_list[i], text_list[i]
                all_texts[all_spks.index(spk)].append(text)
            for i in range(len(all_spks)):
                all_texts[i] = "({}) {}".format(all_spks[i], " ".join(all_texts[i]))
            text_label = '#'.join(all_texts)
            text_file.write("{} {}\n".format(wavname, text_label))
            utt2spk_file.write("{} {}\n".format(wavname, wavname))
            
            for (spk, text, timebound) in zip(spk_list, text_list, timebound_list):
                rttm_file.write("SPEAKER {} 1 {:.2f} {:.2f} <NA> <NA> {} <NA> <NA> {}\n".format(wavname, timebound[0]-start_t, timebound[1] - timebound[0], spk, text))
            
            reco2dur_file.write("{} {:.2f}\n".format(wavname, end_t - start_t))
            
            utt2numspks_file.write("{} {}\n".format(wavname, len(set(spk_list))))
            utt2overlap_file.write("{} {:.2f}\n".format(wavname, compute_overlap_ratio(spk_list, timebound_list, start_t)))

    wav_scp_file.close()
    text_file.close()
    utt2spk_file.close()
    rttm_file.close()
    reco2dur_file.close()
    utt2numspks_file.close()
    utt2overlap_file.close()
    print("Total {}, skipping {} short, {} long".format(cnt, cnt_short, cnt_long))
    return 0

if __name__ == '__main__':
    main()
