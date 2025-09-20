import os
import torch
import torch.nn as nn
import argparse
import numpy as np
from s3prl.downstream.runner import Runner
from typing import Optional
from torch.nn.utils.rnn import pad_sequence
from s3prl.downstream.sep_ami.dataset import SeparationDatasetUttGroup
import soundfile as sf
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Evaluate separation model')
parser.add_argument('ckpt_path', type=str, help='checkpoint path')
parser.add_argument('test_dir', type=str, help='test directory')
parser.add_argument('sdm1_dir', type=str, help='sdm1 directory')
parser.add_argument('output_dir', type=str, help='output directory')
parser.add_argument('--channel', type=str, default='0', help='channels to use')
parser.add_argument('--normalize', type=int, default=0, help='whether to normalize audio')
parser.add_argument('--n_fft', type=int, default=512)
parser.add_argument('--hop_length', type=int, default=160)
parser.add_argument('--win_length', type=int, default=400)
parser.add_argument('--num_srcs', type=int, default=1)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

def compute_energy_dB(src):
    assert len(src.shape) == 1
    return 10 * np.log10(max(1e-20, np.mean(src ** 2)))

class S3PRLModel(nn.Module):
    def __init__(
        self,
        upstream,
        featurizer,
        downstream,
        datarc,
    ):
        super(S3PRLModel, self).__init__()
        self.upstream = upstream
        self.featurizer = featurizer
        self.downstream = downstream
        self.datarc = datarc
        self.ref_channel = datarc['ref_channel'] if 'ref_channel' in datarc else 0
        self.n_fft = datarc['n_fft']
        self.hop_length = datarc['hop_length']
        self.win_length = datarc['win_length']
        self.upstream_rate = self.featurizer.model.downsample_rate
        self.upstream.model.eval()
        self.downstream.model.model.eval()

    def forward(self, waveforms: torch.Tensor, sdm1_waveforms: torch.Tensor) -> torch.Tensor:
        #print("waveforms", waveforms.shape, "sdm1_waveforms", sdm1_waveforms.shape)
        if waveforms.size(2) == 1:
            wavs = [(wav[:, 0]).to(device) for wav in waveforms]
        else:
            wavs = [(wav[:, :]).to(device) for wav in waveforms]
        #print([wav.shape for wav in wavs])
        with torch.no_grad():
            features = self.upstream.model(wavs)
            features = self.featurizer.model(wavs, features)
            features = pad_sequence(features, batch_first=True)
            #print("features", features.shape)

            src_stft = torch.stft(
                sdm1_waveforms.to(device),
                self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=torch.hann_window(self.win_length).to(device),
                return_complex=True,
            )
            src_stft = src_stft.permute(0, 2, 1) 
            src_stft_mag = torch.abs(src_stft) + 1e-10
            src_stft_phase = src_stft / src_stft_mag
            
            assert self.upstream_rate == 160 or self.upstream_rate == 320
            assert self.hop_length == 160
            if self.upstream_rate == 320:
                features = torch.repeat_interleave(features, 2, dim=1)
            # match the length
            min_length = min(features.size(1), src_stft_mag.size(1))
            features = features[:, :min_length, :]
            src_stft_mag, src_stft_phase = src_stft_mag[:, :min_length, :], src_stft_phase[:, :min_length, :] 
            #print("src_stft_mag", src_stft_mag.shape)

            if self.datarc["concatenate"]:
                if self.datarc["log1p"]:
                    input_feature = torch.cat([features, torch.log1p(src_stft_mag)], 2)
                else:
                    input_feature = torch.cat([features, src_stft_mag], 2)
            else:
                input_feature = features
            #print("input_feature", input_feature.shape)

            pred_mask = self.downstream.model.model(input_feature)
            #print("pred_mask", pred_mask.shape)
            #print("pred_mask", torch.max(pred_mask), torch.min(pred_mask))

            pred_stft_mag = src_stft_mag.unsqueeze(1) * pred_mask.permute(0, 2, 1, 3)
            pred_stft = pred_stft_mag * src_stft_phase.unsqueeze(1)
            bs, num_srcs = pred_stft.size(0), pred_stft.size(1)
            pred_audios = torch.istft(
                    pred_stft.view(bs * num_srcs, pred_stft.size(2), pred_stft.size(3)).permute(0, 2, 1), 
                    n_fft=self.datarc["n_fft"],
                    hop_length=self.datarc["hop_length"],
                    win_length=self.datarc["win_length"],
                    window=torch.hann_window(self.datarc["win_length"]).to(device),
                    center=True,
                )
            #print("bs", bs, "num_srcs", num_srcs)
            #print("pred_audios", pred_audios.shape)
            pred_audios = pred_audios.view(bs, num_srcs, pred_audios.size(1))

            # pad the predicted audios to the same length
            pred_audios_pad = torch.zeros(bs, num_srcs, waveforms.size(1))
            pred_audios_pad[:, :, :pred_audios.size(2)] = pred_audios
        return pred_audios_pad[0]

def main():
    if not os.path.exists(args.output_dir + '/data'):
        os.makedirs(args.output_dir + '/data')

    print("Loading {}".format(args.ckpt_path))
    ckpt = torch.load(args.ckpt_path, map_location='cpu')
    ckpt_args, ckpt_config = ckpt['Args'], ckpt['Config']
    datarc = ckpt_config['downstream_expert']['datarc']
    ckpt_args.init_ckpt = args.ckpt_path
    runner = Runner(ckpt_args, ckpt_config)
    upstream = runner.upstream
    featurizer = runner.featurizer
    downstream = runner.downstream
    separation_model = S3PRLModel(upstream, featurizer, downstream, datarc)

    dataset = SeparationDatasetUttGroup(args.test_dir, args.sdm1_dir, rate=16000, channel=args.channel, normalize=args.normalize)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    wav_scp_file = open("{}/wav.scp".format(args.output_dir), 'w')
    utt2num_samples_file = open("{}/utt2num_samples".format(args.output_dir), 'w')
    utt2spk_file = open("{}/utt2spk".format(args.output_dir), 'w')
    for i, v in tqdm(enumerate(dataloader), total=len(dataloader)):
        audio, sdm1_audio, textlist, uttname = v
        audio, sdm1_audio = (audio.to(device)).float(), (sdm1_audio.to(device)).float()
        uttname = uttname[0]
        pred_audios = separation_model(audio, sdm1_audio)
        assert args.num_srcs <= 2 and pred_audios.size(0) == 2

        energy_1 = compute_energy_dB(pred_audios[0, :].data.cpu().numpy())
        energy_2 = compute_energy_dB(pred_audios[1, :].data.cpu().numpy())
        if energy_1 < energy_2:
            pred_audios = torch.flip(pred_audios, dims=[0])

        sf.write("{}/data/{}_mix.wav".format(args.output_dir, uttname), audio[0, :, 0].data.cpu().numpy(), 16000)
        # wav.scp, utt2num_samples, utt2spk
        for src_id in range(args.num_srcs):
            sf.write("{}/data/{}_s{}.wav".format(args.output_dir, uttname, src_id+1), pred_audios[src_id, :].data.cpu().numpy(), 16000)
            wav_scp_file.write("{}_s{} {}/data/{}_s{}.wav\n".format(uttname, src_id+1, args.output_dir, uttname, src_id+1))
            utt2num_samples_file.write("{}_s{} {}\n".format(uttname, src_id+1, audio.size(2)))
            utt2spk_file.write("{}_s{} {}_s{}\n".format(uttname, src_id+1, uttname, src_id+1))
    wav_scp_file.close()
    utt2num_samples_file.close()
    utt2spk_file.close()
    return 0

if __name__ == '__main__':
    main()
