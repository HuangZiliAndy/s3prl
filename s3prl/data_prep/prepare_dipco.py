import os
import soundfile as sf
import argparse
import subprocess
from tqdm import tqdm
import json
import numpy as np

parser = argparse.ArgumentParser(description='Prepare DiPCo dataset (unsegmented)')
parser.add_argument('DiPCo_dir', type=str, help='DiPCo dataset directory')
parser.add_argument('output_dir', type=str, help='Output directory')
parser.add_argument('--cond', type=str, default='SDM1', help='Condition of audio')
parser.add_argument('--merge_dis', type=float, default=0.0, help='Merge distance')
args = parser.parse_args()

def merge_seg(seg_list, merge_dis):
    seg_list_merged = []
    start_t, end_t, text_list = None, None, None
    for i, v in enumerate(seg_list):
        if start_t is None:
            start_t, end_t, text_list = v[0], v[1], [v[2]]
        else:
            if v[0] > end_t + merge_dis:
                seg_list_merged.append([start_t, end_t, ' '.join(text_list), v[3]])
                start_t, end_t, text_list = v[0], v[1], [v[2]]
            else:
                end_t = v[1]
                text_list.append(v[2])
    seg_list_merged.append([start_t, end_t, ' '.join(text_list), v[3]])
    return seg_list_merged

def merge_segments(seg_list, merge_dis):
    spk_list = [seg[3] for seg in seg_list]
    spk_list = list(set(spk_list))
    spk_list.sort()
    seg_list_merged = []
    for spk in spk_list:
        seg_list_spk = [seg for seg in seg_list if seg[3] == spk]
        seg_list_spk = sorted(seg_list_spk, key=lambda x: x[0])
        seg_list_merged += merge_seg(seg_list_spk, merge_dis)
    seg_list_merged = sorted(seg_list_merged, key=lambda x: x[0])
    return seg_list_merged

def get_segments(fname):
    with open(fname, 'r') as fh:
        content = fh.readlines()
    seg_list = []
    for line in content:
        line = line.strip('\n')
        line_split = line.split()
        assert len(line_split) == 4
        start_t, end_t, word, spk = line_split 
        start_t = round(float(start_t), 2)
        end_t = round(float(end_t), 2)
        seg_list.append([start_t, end_t, word, spk])
    seg_list_merged = merge_segments(seg_list, args.merge_dis)
    return seg_list_merged

def load_label(json_file):
    seg_list = []
    with open(json_file, "r") as f:
        label = json.load(f)
    for line in label:
        start_t_list = [convert_time(line['start_time'][mic]) for mic in ["U01", "U02", "U03", "U04", "U05", "close-talk"]]
        end_t_list = [convert_time(line['end_time'][mic]) for mic in ["U01", "U02", "U03", "U04", "U05", "close-talk"]]
        assert np.max(start_t_list) == np.min(start_t_list) and np.max(end_t_list) == np.min(end_t_list)
        start_t, end_t = np.max(start_t_list), np.max(end_t_list)
        word_list = line['words'].split()
        word_list = [w for w in word_list if w not in ['[unintelligible]', '[laugh]', '[noise]']]
        seg_list.append([start_t, end_t, ' '.join(word_list), line['speaker_id']])
    seg_list_merged = merge_segments(seg_list, args.merge_dis)
    return seg_list_merged

def convert_time(time_str):
    h, m, s = time_str.split(":")
    seconds = int(h) * 3600 + int(m) * 60 + float(s)
    return seconds

def main():
    for split in ["dev", "eval"]:
        audio_dir = "{}/audio/{}".format(args.DiPCo_dir, split)
        text_dir = "{}/transcriptions/{}".format(args.DiPCo_dir, split)

        output_dir = "{}/{}/{}".format(args.output_dir, args.cond, split)
        wav_dir = "{}/wav".format(output_dir)
        if not os.path.exists(wav_dir):
            os.makedirs(wav_dir)
        rttm_dir = "{}/rttm".format(output_dir)
        if not os.path.exists(rttm_dir):
            os.makedirs(rttm_dir)

        wav_scp_file = open("{}/wav.scp".format(output_dir), 'w')
        utt2spk_file = open("{}/utt2spk".format(output_dir), 'w')
        reco2dur_file = open("{}/reco2dur".format(output_dir), 'w')
        segments_file = open("{}/segments".format(output_dir), 'w')
        text_file = open("{}/text".format(output_dir), 'w')
        uem_file = open("{}/uem".format(output_dir), 'w')
        rttm_scp_file = open("{}/rttm.scp".format(output_dir), 'w')

        sessions = list(set([f.split('_')[0] for f in os.listdir(audio_dir) if f.endswith('.wav') and f.startswith('S')]))
        print("{} meetings in split {}".format(len(sessions), split))
        sessions.sort()


        for sess_name in tqdm(sessions):
            duration_list = []
            json_file = "{}/transcriptions/{}/{}.json".format(args.DiPCo_dir, split, sess_name)
            assert os.path.exists(json_file)
            segments = load_label(json_file)
            for mic in ["U01", "U02", "U03", "U04", "U05"]: 
                meet_name = "{}_{}".format(sess_name, mic)
                audio_file = "{}/audio/{}/{}.CH7.wav".format(args.DiPCo_dir, split, meet_name)
                assert os.path.exists(audio_file)

                duration = sf.info(audio_file).duration
                channels = sf.info(audio_file).channels
                reco2dur_file.write("{} {}\n".format(meet_name, duration))
                uem_file.write("{} 1 {} {}\n".format(meet_name, 0, duration))

                if args.cond == "MDM":
                    raise
                elif args.cond == "SDM1":
                    wav_scp_file.write("{} {}\n".format(meet_name, audio_file))
                else:
                    raise ValueError("Condition not defined.")

                rttm_filename = "{}/{}.rttm".format(rttm_dir, meet_name)
                rttm_file = open(rttm_filename, 'w')
                for i, seg in enumerate(segments):
                    start_t, end_t = round(seg[0], 2), round(seg[1], 2)
                    text, spkname = seg[2].lower(), seg[3]
                    segment_name = "{}_{}_{:07d}_{:07d}".format(meet_name, spkname, int(100.0 * start_t), int(100.0 * end_t))
                    segments_file.write("{} {} {:.2f} {:.2f}\n".format(segment_name, meet_name, start_t, end_t))
                    utt2spk_file.write("{} {}\n".format(segment_name, spkname))
                    text_file.write("{} {}\n".format(segment_name, text))
                    rttm_file.write("SPEAKER {} 1 {:.2f} {:.2f} <NA> <NA> {} <NA> <NA>\n".format(meet_name, start_t, end_t - start_t, spkname))
                rttm_file.close()

                rttm_scp_file.write("{} {}\n".format(meet_name, rttm_file))

        wav_scp_file.close()
        utt2spk_file.close()
        reco2dur_file.close()
        segments_file.close()
        text_file.close() 
        uem_file.close()
        rttm_scp_file.close()
    return 0

if __name__ == '__main__':
    main()
