import os
import json
import soundfile as sf
import argparse
import textgrid
import subprocess

parser = argparse.ArgumentParser(description='Prepare CHiME6 dataset (unsegmented)')
parser.add_argument('CHiME6_dir', type=str, help='CHiME6 dataset directory')
parser.add_argument('output_dir', type=str, help='Output directory')
parser.add_argument('--cond', type=str, default='SDM1', help='Condition of audio')
args = parser.parse_args()

def time_to_seconds(time_str):
    hours, minutes, seconds = map(float, time_str.split(':'))
    
    total_seconds = hours * 3600 + minutes * 60 + seconds
    return total_seconds

def load_label(fname, session):
    seg_list = []
    with open(fname, 'r') as fh:
        json_data = json.load(fh)
    for item in json_data:
        for attr in ['end_time', 'start_time', 'words', 'speaker']:
            assert attr in item
        start_t, end_t = time_to_seconds(item['start_time']), time_to_seconds(item['end_time'])
        start_t, end_t = round(start_t, 2), round(end_t, 2)
        spk = item['speaker']
        seg_list.append([session, start_t, end_t, spk, item['words']])
    return seg_list

def main():
    for split in ["dev", "eval", "train"]:
        audio_dir = "{}/audio/{}".format(args.CHiME6_dir, split)
        label_dir = "{}/transcriptions/{}".format(args.CHiME6_dir, split)
        label_files = os.listdir(label_dir)
        label_files.sort()
        session_list = [label_file.strip('.json') for label_file in label_files]
        print("{} split, {} sessions".format(split, len(session_list)))

        output_dir = "{}/{}/{}".format(args.output_dir, args.cond, split)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        wav_scp_file = open("{}/wav.scp".format(output_dir), 'w')
        reco2dur_file = open("{}/reco2dur".format(output_dir), 'w')
        segments_file = open("{}/segments".format(output_dir), 'w')

        for session in session_list:
            label_file = "{}/{}.json".format(label_dir, session)
            seg_list = load_label(label_file, session)
            audio_list = [f for f in os.listdir(audio_dir) if f.startswith(session)]
            audio_list.sort()
            if args.cond == 'SDM1':
                audio_list = [f for f in audio_list if 'CH1' in f]
            else:
                raise NotImplementedError
            for audio in audio_list:
                audio_path = "{}/{}".format(audio_dir, audio)
                assert os.path.exists(audio_path)
                duration = sf.info(audio_path).frames / sf.info(audio_path).samplerate
                wav_scp_file.write("{} {}\n".format(audio.split('.')[0], audio_path))
                reco2dur_file.write("{} {}\n".format(audio.split('.')[0], duration))
                for seg in seg_list:
                    segname = "{}-{}-{:07d}-{:07d}".format(seg[3], audio.split('.')[0], int(100.0 * seg[1]), int(100.0 * seg[2]))
                    segments_file.write("{} {} {:.2f} {:.2f}\n".format(segname, audio.split('.')[0], seg[1], seg[2]))
        segments_file.close()
        reco2dur_file.close()
        segments_file.close()
    return 0

if __name__ == '__main__':
    main()
