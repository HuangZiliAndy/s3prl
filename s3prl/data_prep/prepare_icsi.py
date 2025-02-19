import os
import soundfile as sf
import argparse
import subprocess
from lhotse.recipes import prepare_icsi

#parser = argparse.ArgumentParser(description='Prepare ICSI dataset (unsegmented)')
#parser.add_argument('ICSI_dir', type=str, help='ICSI dataset directory')
#parser.add_argument('output_dir', type=str, help='Output directory')
#parser.add_argument('--cond', type=str, default='SDM1', help='Condition of audio')
#args = parser.parse_args()



def main():
    icsi = prepare_icsi(
        audio_dir="/export/corpora5/LDC/LDC2004S02/meeting_speech",
        transcripts_dir="/export/c02/hzili1/datasets/ICSI_annotation/transcripts",
        output_dir="/export/c02/hzili1/datasets/s3prl_csp/debug",
        mic="sdm",
        normalize_text="kaldi",
        save_to_wav=False,
    )
    recordings = icsi['recordings']
    supervisions = icsi['supervisions']
    print("len(recordings)", len(recordings))
    print("len(supervisions)", len(supervisions))
    print("len(recordings)", recordings.keys())
    return 0

if __name__ == '__main__':
    main()
