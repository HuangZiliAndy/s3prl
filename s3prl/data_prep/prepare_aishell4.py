import os
import soundfile as sf
import argparse
import textgrid
import subprocess
import re

parser = argparse.ArgumentParser(description='Prepare AISHELL4 dataset (unsegmented)')
parser.add_argument('AISHELL_dir', type=str, help='AISHELL4 dataset directory')
parser.add_argument('output_dir', type=str, help='Output directory')
parser.add_argument('--cond', type=str, default='SDM1', help='Condition of audio')
args = parser.parse_args()

def text_normalize(line: str) -> str:
    line = line.replace("<sil>", "")
    line = line.replace("<%>", "")
    line = line.replace("<->", "")
    line = line.replace("<$>", "")
    line = line.replace("<#>", "")
    line = line.replace("<$>", "")
    line = line.replace("<_>", "")
    line = line.replace("<space>", "")
    line = line.replace("`", "")
    line = line.replace("&", "")
    line = line.replace(",", "")
    line = line.replace("\r", "")
    line = line.replace("\n", "")
    if re.search("[a-zA-Z]", line):
        line = line.upper()
    line = line.replace("Ａ", "A")
    line = line.replace("ａ", "A")
    line = line.replace("ｂ", "B")
    line = line.replace("ｃ", "C")
    line = line.replace("ｋ", "K")
    line = line.replace("ｔ", "T")
    line = line.replace("，", "")
    line = line.replace("丶", "")
    line = line.replace("。", "")
    line = line.replace("、", "")
    line = line.replace("？", "")
    return line

def main():
    #for split in ["test", "train_L", "train_M", "train_S"]:
    #for split in ["test", "train_M", "train_S"]:
    for split in ["train_M"]:
        split_dir="{}/{}".format(args.AISHELL_dir, split) 

        assert os.path.exists("{}/TextGrid".format(split_dir)) and os.path.exists("{}/wav".format(split_dir))

        output_dir = "{}/{}/{}".format(args.output_dir, args.cond, split)

        if not os.path.exists("{}/wav".format(output_dir)):
            os.makedirs("{}/wav".format(output_dir))

        audio_files = [f for f in os.listdir("{}/wav".format(split_dir)) if f.endswith('.wav')]
        textgrid_files = [f for f in os.listdir("{}/TextGrid".format(split_dir)) if f.endswith('.TextGrid')]
        print("{} meetings in split {}".format(len(audio_files), split))
        audio_files.sort()

        wav_scp_file = open("{}/wav.scp".format(output_dir), 'w')
        reco2dur_file = open("{}/reco2dur".format(output_dir), 'w')
        segments_file = open("{}/segments".format(output_dir), 'w')
        text_file = open("{}/text".format(output_dir), 'w')

        for i, audio_file in enumerate(audio_files):
            print("{} / {}".format(i+1, len(audio_files)))

            meet_name = audio_file.strip('.wav')

            audio_file = "{}/wav/{}".format(split_dir, audio_file)
            textgrid_file = meet_name + '.TextGrid'
            textgrid_file = "{}/TextGrid/{}".format(split_dir, textgrid_file)
            assert os.path.exists(audio_file) and os.path.exists(textgrid_file)

            try:
                tg = textgrid.TextGrid.fromFile(textgrid_file)
            except:
                print("Error processing {}".format(textgrid_file))
                continue

            if args.cond == 'SDM1':
                audio, sr = sf.read(audio_file)
                assert sr == 16000 and audio.shape[1] == 8
                sf.write("{}/wav/{}.wav".format(output_dir, meet_name), audio[:, 0], sr)
                duration = len(audio) / sr
                wav_scp_file.write("{} {}/wav/{}.wav\n".format(meet_name, output_dir, meet_name))
                reco2dur_file.write("{} {}\n".format(meet_name, duration))
            else:
                raise NotImplementedError

            for tier in tg.tiers:
                spkname = tier.name
                for i, interval in enumerate(tier.intervals):
                    start_t, end_t = interval.minTime, interval.maxTime
                    assert start_t < end_t
                    text = text_normalize(interval.mark)
                    if text != "":
                        if end_t <= duration:
                            start_t, end_t = round(start_t, 2), round(end_t, 2)
                            segname = "{}-{}-{:07d}-{:07d}".format(spkname, meet_name, int(100.0 * start_t), int(100.0 * end_t))
                            segments_file.write("{} {} {} {}\n".format(segname, meet_name, start_t, end_t))
                            text_file.write("{} {}\n".format(segname, text))

        wav_scp_file.close()
        reco2dur_file.close()
        segments_file.close()
        text_file.close()
    return 0

if __name__ == '__main__':
    main()
