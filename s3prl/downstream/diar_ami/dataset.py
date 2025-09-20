import io
import os
import numpy as np
import torch
import soundfile as sf
import subprocess
from torch.utils.data.dataset import Dataset

class DiarizationDataset(Dataset):
    def __init__(
            self,
            data_dir, 
            num_spks,
            frame_shift,
            channel: str = "0",
            normalize: bool = False,
            **kwargs
        ):
        self.data_dir = data_dir
        for f in ["wav.scp", "rttm.scp", "reco2dur"]:
            os.path.exists("{}/{}".format(data_dir, f))
        self.utt2path = self.load_scp("{}/wav.scp".format(data_dir))
        self.utt2rttm = self.load_scp("{}/rttm.scp".format(data_dir))
        self.reco2dur = self.load_scp("{}/reco2dur".format(data_dir))
        assert len(self.utt2path) == len(self.utt2rttm) and len(self.utt2path) == len(self.reco2dur)
        self.uttlist = list(self.utt2path.keys())
        self.uttlist.sort()
        self.num_spks = num_spks
        self.frame_rate = frame_shift / 16000.0
        self.channel = [int(c) for c in channel.split(',')]
        if len(self.channel) == 1:
            self.channel = self.channel[0]
        self.normalize = normalize

    def load_scp(self, fname):
        utt2path = {}
        with open(fname, 'r') as fh:
            content = fh.readlines()
        for line in content:
            line = line.strip('\n')
            line_split = line.split(None, 1)
            utt, path = line_split[0], line_split[1]
            utt2path[utt] = path
        return utt2path

    def __len__(self):
        return len(self.uttlist)

    def get_audio(self, wav_path):
        if wav_path.endswith("|"):
            # input piped command
            p = subprocess.Popen(
                wav_path[:-1],
                shell=True,
                stdout=subprocess.PIPE,
            )
            audio, samplerate = sf.read(
                io.BytesIO(p.stdout.read()),
                dtype="float32",
            )
        else:
            audio, samplerate = sf.read(wav_path)
        return audio

    def get_label(self, rttm_file, duration):
        nframes = int(np.round(duration / self.frame_rate))
        label = np.zeros((nframes, self.num_spks))
        with open(rttm_file, 'r') as fh:
            content = fh.readlines()
        seg_list = []
        for line in content:
            line = line.strip('\n')
            line_split = line.split()
            start_t, end_t, spk = float(line_split[3]), float(line_split[3]) + float(line_split[4]), line_split[7]
            start_t, end_t = round(start_t, 2), round(end_t, 2)
            seg_list.append([start_t, end_t, spk])
        spk_list = list(set([seg[2] for seg in seg_list]))
        spk_list.sort()
        assert len(spk_list) <= self.num_spks
        for seg in seg_list:
            start_frame, end_frame = int(round(seg[0] / self.frame_rate)), int(round(seg[1] / self.frame_rate))
            spk_idx = spk_list.index(seg[2])
            label[start_frame:end_frame, spk_idx] = 1
        return label, seg_list

    def __getitem__(self, index):
        uttname = self.uttlist[index]
        wav_path = self.utt2path[uttname] 
        rttm_path = self.utt2rttm[uttname]
        duration = float(self.reco2dur[uttname])
        audio = self.get_audio(wav_path)
        if self.normalize:
            audio = audio / np.max(np.abs(audio))
        if len(audio.shape) == 2:
            audio = audio[:, self.channel]
        elif len(audio.shape) == 1:
            assert self.channel == 0
        else:
            raise ValueError("Invalid audio shape")
        label, _ = self.get_label(rttm_path, duration) 
        return audio, label, label.shape[0], uttname

    def collate_fn(self, samples):
        return zip(*samples)

if __name__ == '__main__':
    data_dir = "/export/c02/hzili1/datasets/s3prl_csp/downstream/diar_ami/SDM1/train"
    num_spks = 5
    frame_shift = 320
    dataset = DiarizationDataset(
            data_dir,
            num_spks,
            frame_shift
        )
    for i, v in enumerate(dataset):
        audio, label, length, uttname = v
        print('-' * 80)
        print(uttname)
        print("audio", audio.shape)
        print("label", label.shape)
        print("label", np.sum(label, 0))
        print("length", length)
        break
