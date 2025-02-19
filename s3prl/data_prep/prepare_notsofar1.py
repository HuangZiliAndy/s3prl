import os
import soundfile as sf
import argparse
import textgrid
import subprocess
import json

parser = argparse.ArgumentParser(description='Prepare NOTSOFAR dataset (unsegmented)')
parser.add_argument('NSF_dir', type=str, help='NOTSOFAR dataset directory')
parser.add_argument('output_dir', type=str, help='Output directory')
parser.add_argument('--cond', type=str, default='SDM1', help='Condition of audio')
args = parser.parse_args()

def get_label(fname):
    seg_list = []
    with open(fname, 'r') as fh:
        label = json.load(fh)
    for tup in label:
        seg_list.append([tup['start_time'], tup['end_time'], tup['speaker_id'], tup['text']])
    return seg_list

def main():
    for split in ["train", "dev", "eval"]:
        if split == 'train':
            split_dir="{}/train_set/240825.1_train/MTG".format(args.NSF_dir)
        elif split == 'dev':
            split_dir="{}/dev_set/240415.2_dev_with_GT/MTG".format(args.NSF_dir)
        elif split == 'eval':
            split_dir="{}/eval_set/240629.1_eval_small_with_GT/MTG".format(args.NSF_dir)

        meeting_list = os.listdir(split_dir) 
        meeting_list.sort()
        print("{} split, {} meetings".format(split, len(meeting_list)))
        
        output_dir = "{}/{}/{}".format(args.output_dir, args.cond, split)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        wav_scp_file = open("{}/wav.scp".format(output_dir), 'w')
        reco2dur_file = open("{}/reco2dur".format(output_dir), 'w')
        segments_file = open("{}/segments".format(output_dir), 'w')
        text_file = open("{}/text".format(output_dir), 'w')

        for i, meet in enumerate(meeting_list):
            print("{} / {}".format(i+1, len(meeting_list)))
            anno_file = "{}/{}/gt_transcription.json".format(split_dir, meet)
            assert os.path.exists(anno_file)
            seg_list = get_label(anno_file)
            
            mc_dir_list = [path for path in os.listdir("{}/{}".format(split_dir, meet)) if path.startswith('mc')]
            mc_dir_list.sort()

            n_channels, rec_nsamples = [], []
            for mc_dir in mc_dir_list:
                full_path = "{}/{}/{}".format(split_dir, meet, mc_dir)
                assert os.path.exists("{}/ch0.wav".format(full_path))
                n_channels.append(len(os.listdir(full_path)))
                rec_nsamples.append(sf.info("{}/ch0.wav".format(full_path)).frames)

                if args.cond == 'SDM1':
                    wavname = "{}_{}".format(meet, '_'.join(mc_dir.split('_')[1:]))
                    wavfile = "{}/ch0.wav".format(full_path)
                    duration = sf.info(wavfile).frames / sf.info(wavfile).samplerate
                    wav_scp_file.write("{} {}\n".format(wavname, wavfile))
                    reco2dur_file.write("{} {}\n".format(wavname, duration))

                for seg in seg_list:
                    wavname = "{}_{}".format(meet, '_'.join(mc_dir.split('_')[1:]))
                    segname = "{}-{}-{:07d}-{:07d}".format(seg[2], wavname, int(100.0 * seg[0]), int(100.0 * seg[1]))
                    segments_file.write("{} {} {} {}\n".format(segname, wavname, seg[0], seg[1]))
                    text_file.write("{} {}\n".format(segname, seg[3]))
            assert max(rec_nsamples) == min(rec_nsamples)
            assert min(n_channels) > 1

        wav_scp_file.close()
        reco2dur_file.close()
        segments_file.close()
        text_file.close()
    return 0

if __name__ == '__main__':
    main()
