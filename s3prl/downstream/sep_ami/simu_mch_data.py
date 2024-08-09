import os
import math
import numpy as np
import pyroomacoustics as pra
import matplotlib.pyplot as plt
import soundfile as sf
from scipy import signal
import argparse
import time
import h5py
from pathlib import Path
from s3prl.downstream.sep_ami.noise import * 

parser = argparse.ArgumentParser(description='Generate multichannel audios')
parser.add_argument('src_dir', type=str, help='source directory')
parser.add_argument('output_dir', type=str, help='output directory')
parser.add_argument('--noise_scp_file', type=str, help='noise scp file')
parser.add_argument('--RIR_dir', type=str, help='RIR directory')
parser.add_argument('--num_spk', type=str, default='2', help='number of speakers')
parser.add_argument('--num_spk_prob', type=str, default='1.0', help='number of speakers probability')
parser.add_argument('--add_noise', type=int, default=0, help='whether to add noise')
parser.add_argument('--noise_type', type=str, default='diffuse', help='type of noises')
parser.add_argument('--add_reverb', type=int, default=0, help='whether to add reverberation')
parser.add_argument('--sr', type=int, default=16000, help='sample rate')
parser.add_argument('--sir_range', type=str, default='-5,5', help='range of SIR')
parser.add_argument('--snr_range', type=str, default='5,20', help='range of SNR')
parser.add_argument('--crop_dur', type=float, default=10.0, help='crop utterance if longer than crop_dur')
parser.add_argument('--num_utts', type=int, default=20000, help='number of utts to generate')
parser.add_argument('--normalize', type=int, default=0, help='whether to normalize audio')
parser.add_argument('--s1_first', type=int, default=1, help='s1 first in the mixture')
parser.add_argument('--s1_only', type=int, default=0, help='the start and end sample same as s1')
parser.add_argument('--full_overlap', type=int, default=0, help='create mixture with full overlap')
parser.add_argument('--early_rir_dur', type=float, default=0.05, help='duration of the direct wave (default 50ms)')
parser.add_argument('--single_channel', type=int, default=0, help='whether to simulate single channel speech')
parser.add_argument('--seed', type=int, default=7, help='random seed')
args = parser.parse_args()

def compute_snr(signal, noise):
    sig_pwr = np.mean(signal**2)
    noz_pwr = np.mean(noise**2)
    if sig_pwr == 0.0:
        return -np.inf
    elif noz_pwr == 0.0:
        return np.inf
    else:
        return 10.0 * np.log10(sig_pwr / noz_pwr)

# Code from https://github.com/fgnt/sms_wsj
def get_rir_start_sample(h, level_ratio=1e-1):
    """Finds start sample in a room impulse response.

    Selects that index as start sample where the first time
    a value larger than `level_ratio * max_abs_value`
    occurs.

    If you intend to use this heuristic, test it on simulated and real RIR
    first. This heuristic is developed on MIRD database RIRs and on some
    simulated RIRs but may not be appropriate for your database.

    If you want to use it to shorten impulse responses, keep the initial part
    of the room impulse response intact and just set the tail to zero.

    Params:
        h: Room impulse response with Shape (num_samples,)
        level_ratio: Ratio between start value and max value.

    >>> get_rir_start_sample(np.array([0, 0, 1, 0.5, 0.1]))
    2
    """
    assert level_ratio < 1, level_ratio
    if h.ndim > 1:
        assert h.shape[0] < 20, h.shape
        h = np.reshape(h, (-1, h.shape[-1]))
        start_sample_list = [get_rir_start_sample(h_, level_ratio=level_ratio) for h_ in h]
        return np.min(start_sample_list)

    abs_h = np.abs(h)
    max_index = np.argmax(abs_h)
    max_abs_value = abs_h[max_index]
    # +1 because python excludes the last value
    larger_than_threshold = abs_h[:max_index + 1] > level_ratio * max_abs_value

    # Finds first occurrence of max
    rir_start_sample = np.argmax(larger_than_threshold)
    return rir_start_sample

def compute_energy_dB(src):
    #assert len(src.shape) == 1
    return 10 * np.log10(max(1e-20, np.mean(src ** 2)))

def scale_audios(srcs, snr_list):
    energy_dB_list = [compute_energy_dB(src) for src in srcs]
    gain_list = [min(40, -snr_list[i]+energy_dB_list[0]-energy_dB_list[i]) for i in range(len(energy_dB_list))]
    scale_srcs = [srcs[i] * np.power(10, (gain_list[i] / 20.)) for i in range(len(srcs))]
    #energy_dB_list = [compute_energy_dB(src) for src in scale_srcs]
    return scale_srcs, gain_list

def mch_rir_conv(input_wav, mch_rir, early_rir_samples):
    input_wav = np.expand_dims(input_wav, axis=0)
    start_idx = np.argmax(mch_rir[0])
    #start_idx = get_rir_start_sample(mch_rir)
    end_idx_direct = min(start_idx + early_rir_samples, mch_rir.shape[1])

    mch_rir_early = mch_rir.copy()
    mch_rir_early[:, end_idx_direct:] = 0

    #start_idx = np.argmax(mch_rir[0])
    reverb_wav = signal.fftconvolve(input_wav, mch_rir, mode="full")
    reverb_wav = reverb_wav[:, start_idx:start_idx + input_wav.shape[-1]]
    direct_wav = signal.fftconvolve(input_wav, mch_rir_early, mode="full")
    direct_wav = direct_wav[:, start_idx:start_idx + input_wav.shape[-1]]
    return reverb_wav, direct_wav

def convert_seconds(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = (seconds % 3600) % 60
    return hours, minutes, seconds

def parse_data_dir(src_dir):
    assert os.path.exists("{}/wav.scp".format(src_dir)) and os.path.exists("{}/reco2dur".format(src_dir))
    utt2path = get_wav_scp("{}/wav.scp".format(src_dir))
    reco2dur = get_reco2dur("{}/reco2dur".format(src_dir))
    uttlist = list(utt2path.keys())
    uttlist.sort()
    location2dur, location2spk, spk2dur, spk2utt = {}, {}, {}, {}
    for utt in uttlist:
        location = utt[0]
        spk = utt.split('_')[1]
        duration = reco2dur[utt]
        if location not in location2dur:
            location2dur[location] = 0
        if location not in location2spk:
            location2spk[location] = []
        if spk not in spk2dur:
            spk2dur[spk] = 0
        if spk not in spk2utt:
            spk2utt[spk] = []
        location2dur[location] += duration
        location2spk[location].append(spk)
        spk2dur[spk] += duration
        spk2utt[spk].append(utt)
    for loc in location2spk.keys():
        spklist = list(set(location2spk[loc]))
        spklist.sort()
        location2spk[loc] = spklist
    return utt2path, reco2dur, location2dur, location2spk, spk2dur, spk2utt 

def get_wav_scp(fname):
    utt2path = {}
    with open(fname, 'r') as fh:
        content = fh.readlines()
    for line in content:
        line = line.strip('\n')
        line_split = line.split()
        utt, path = line_split[0], line_split[1]
        utt2path[utt] = path
    return utt2path

def get_utt2spk(fname):
    utt2spk, spk2utt = {}, {}
    with open(fname, 'r') as fh:
        content = fh.readlines()
    for line in content:
        line = line.strip('\n')
        line_split = line.split()
        utt, spk = line_split[0], line_split[1]
        utt2spk[utt] = spk
        if spk not in spk2utt:
            spk2utt[spk] = []
        spk2utt[spk].append(utt)
    return utt2spk, spk2utt

def get_reco2dur(fname):
    reco2dur = {}
    with open(fname, 'r') as fh:
        content = fh.readlines()
    for line in content:
        line = line.strip('\n')
        line_split = line.split()
        utt, dur = line_split[0], float(line_split[1])
        reco2dur[utt] = dur
    return reco2dur

def get_noise_scp(fname):
    noise_list = []
    with open(fname, 'r') as fh:
        content = fh.readlines()
    for line in content:
        line = line.strip('\n')
        line_split = line.split()
        utt, path = line_split[0], line_split[1]
        noise_list.append(path)
    return noise_list

def align_audios(clean_srcs_list, add_noise, full_overlap, s1_first):
    new_clean_srcs_list = []
    num_samples = [src.shape[0] for src in clean_srcs_list]
    if add_noise:
        num_speech_srcs = len(clean_srcs_list) - 1
        noise = clean_srcs_list[-1]
    else:
        num_speech_srcs = len(clean_srcs_list)
        noise = None

    if num_speech_srcs == 1:
        total_sample = num_samples[0]
        new_clean_srcs_list.append(clean_srcs_list[0])
        start_sample = [0]
    elif num_speech_srcs == 2:
        if full_overlap:
            start_s1, start_s2 = 0, 0
        else:
            if s1_first:
                start_s1 = 0
                start_s2 = np.random.randint(low=0, high=num_samples[0])
            else:
                if np.random.randint(2):
                    start_s1 = 0
                    start_s2 = np.random.randint(low=0, high=num_samples[0])
                else:
                    start_s1 = np.random.randint(low=0, high=num_samples[1])
                    start_s2 = 0
        total_sample = max(start_s1 + num_samples[0], start_s2 + num_samples[1])
        s1, s2 = np.zeros(total_sample,), np.zeros(total_sample,)
        s1[start_s1:start_s1 + num_samples[0]] = clean_srcs_list[0]
        s2[start_s2:start_s2 + num_samples[1]] = clean_srcs_list[1]
        new_clean_srcs_list.append(s1)
        new_clean_srcs_list.append(s2)
        start_sample = [start_s1, start_s2]

    if add_noise:
        if noise.shape[0] <= total_sample:
            noise = np.repeat(noise, int(total_sample / noise.shape[0]) + 1)
        start_noise = np.random.randint(low=0, high=noise.shape[0] - total_sample)
        new_clean_srcs_list.append(noise[start_noise:start_noise + total_sample])
        start_sample.append(0)
    assert len(new_clean_srcs_list) == len(start_sample)

    length_list = [src.shape[0] for src in new_clean_srcs_list]
    assert np.min(length_list) == np.max(length_list)
    return new_clean_srcs_list, start_sample 

def get_RIRs(RIR_dir):
    RIR_files = ["{}/{}".format(RIR_dir, f) for f in os.listdir(RIR_dir)]
    RIR_files.sort()
    return RIR_files

def main():
    print(args)

    np.random.seed(args.seed)

    # load AMI clean segments
    #utt2path, reco2dur, location2dur, location2spk, spk2dur, spk2utt = parse_data_dir(args.src_dir)
    utt2path = get_wav_scp("{}/wav.scp".format(args.src_dir))
    utt2spk, spk2utt = get_utt2spk("{}/utt2spk".format(args.src_dir))
    utt2dur = get_reco2dur("{}/reco2dur".format(args.src_dir))
    spk2dur = {spk : np.sum([utt2dur[utt] for utt in spk2utt[spk]]) for spk in spk2utt.keys()}
    assert len(utt2path) == len(utt2spk) == len(utt2dur)
    spk_list = list(spk2utt.keys())
    spk_list.sort()
    spk_dur_list = [spk2dur[spk] for spk in spk_list]
    print("{} clean AMI segments, {} speakers".format(len(utt2path), len(spk2utt)))
    early_rir_samples = int(args.early_rir_dur * args.sr)

    # load noise segments
    noise_list = get_noise_scp(args.noise_scp_file)
    print("{} noise files".format(len(noise_list)))

    # load RIRs
    if args.RIR_dir:
        RIR_list = get_RIRs(args.RIR_dir)
        print("{} RIRs".format(len(RIR_list)))
    else:
        RIR_list = None

    sir_range = [float(s) for s in args.sir_range.split(',')]
    snr_range = [float(s) for s in args.snr_range.split(',')]
    sir_range.sort()
    snr_range.sort()
    num_spk_list = [int(s) for s in args.num_spk.split(',')]
    num_spk_prob = [float(s) for s in args.num_spk_prob.split(',')]
    
    total_srcs = np.max(num_spk_list) + args.add_noise
    for i in range(total_srcs):
        if i == total_srcs - 1:
            if args.add_noise:
                srcname = "noise"
            else:
                srcname = "s{}".format(i+1)
        else:
            srcname = "s{}".format(i+1)
        if not os.path.exists("{}/wav/{}".format(args.output_dir, srcname)):
            os.makedirs("{}/wav/{}".format(args.output_dir, srcname))
        if not os.path.exists("{}/wav/{}_direct".format(args.output_dir, srcname)):
            os.makedirs("{}/wav/{}_direct".format(args.output_dir, srcname))
    if not os.path.exists("{}/wav/mix".format(args.output_dir)):
        os.makedirs("{}/wav/mix".format(args.output_dir))
    if not os.path.exists("{}/metadata".format(args.output_dir)):
        os.makedirs("{}/metadata".format(args.output_dir))

    wav_scp_file = open("{}/wav.scp".format(args.output_dir), 'w')
    reco2dur_file = open("{}/reco2dur".format(args.output_dir), 'w')
    utt2spk_file = open("{}/utt2spk".format(args.output_dir), 'w')

    for utt_idx in range(1, args.num_utts + 1):
        spk_prob = np.array(spk_dur_list) / np.sum(spk_dur_list)
        num_spk = np.random.choice(num_spk_list, p=num_spk_prob)
        selected_spks = np.random.choice(spk_list, size=num_spk, replace=False, p=spk_prob)
        utt_path_list, snr_list = [], []
        for spk in selected_spks:
            utt_list = spk2utt[spk]
            utt_dur_list = [utt2dur[utt] for utt in utt_list]
            utt_prob = np.array(utt_dur_list) / np.sum(utt_dur_list)
            utt = np.random.choice(utt_list, p=utt_prob)
            utt_path_list.append(utt2path[utt])

        clean_srcs_list, snr_list = [], []
        for i, utt in enumerate(utt_path_list):
            audio, _ = sf.read(utt)
            assert len(audio.shape) == 1
            duration = audio.shape[0] / 16000.0
            if duration <= args.crop_dur:
                clean_srcs_list.append(audio)
            else:
                crop_len = np.random.uniform(low=3.0, high=10.0)
                crop_len = round(crop_len, 2)
                crop_sample = int(16000.0 * crop_len)
                start_sample = np.random.randint(low=0, high=audio.shape[0]-crop_sample)
                clean_srcs_list.append(audio[start_sample:start_sample + crop_sample])
            snr_list.append(0 if i == 0 else np.random.uniform(sir_range[0], sir_range[1]))

        if args.add_noise:
            # randomly select noise file
            noise_file = np.random.choice(noise_list)
            noise, sr = sf.read(noise_file)
            assert sr == 16000
            if len(noise.shape) == 1:
                noise_channel = 0
            elif len(noise.shape) == 2:
                noise_channel = int(np.random.randint(noise.shape[1]))
                noise = noise[:, noise_channel]
            noise_dur = noise.shape[0] / 16000.0
            noise_crop_len = 20.0
            if noise_dur > noise_crop_len:
                crop_sample = int(16000.0 * noise_crop_len)
                start_sample = np.random.randint(low=0, high=noise.shape[0]-crop_sample)
                noise = noise[start_sample:start_sample+crop_sample]
            clean_srcs_list.append(noise)
            snr_list.append(np.random.uniform(snr_range[0], snr_range[1]))
            utt_path_list.append(noise_file)
        assert len(clean_srcs_list) == len(snr_list)

        ## Scale the audios according to the SNR
        #clean_srcs_list = scale_audios(clean_srcs_list, snr_list)

        # Decide the start time of each clean segments
        clean_srcs_list_align, start_sample = align_audios(clean_srcs_list, add_noise=args.add_noise, full_overlap=args.full_overlap, s1_first=args.s1_first)

        if args.s1_only:
            start_sample_s1, end_sample_s1 = start_sample[0], start_sample[0] + clean_srcs_list[0].shape[0] 
            clean_srcs_list_align = [s[start_sample_s1:end_sample_s1] for s in clean_srcs_list_align]

        #clean_srcs = np.stack(clean_srcs_list_align, 0)
        duration = clean_srcs_list_align[0].shape[0] / 16000.0
    
        if args.add_reverb:
            RIR_file = np.load(RIR_list[utt_idx - 1])
            info_dict = {}
            for k in RIR_file.files:
                info_dict[k] = RIR_file[k]
            rir = info_dict['rir']
            if args.single_channel:
                # randomly select one channel
                select_channel = np.random.randint(0, len(rir))
                rir = rir[select_channel:select_channel+1, :, :]

            info_dict['snr'] = np.array(snr_list)
            info_dict['start_sample'] = np.array(start_sample)
            info_dict['uttpath'] = utt_path_list

            reverb_srcs_list, direct_srcs_list = [], []
            for i in range(num_spk):
                reverb_src, direct_src = mch_rir_conv(clean_srcs_list_align[i], rir[:, i, :], early_rir_samples)
                reverb_srcs_list.append(reverb_src)
                direct_srcs_list.append(direct_src)

            if args.add_noise:
                if args.single_channel:
                    assert args.noise_type == 'none'
                if args.noise_type == 'diffuse':
                    diffuse = DiffuseNoise(snr=0, signal=clean_srcs_list_align[-1])
                    noise = diffuse.generate_noise(info_dict['mic_pos'].T, args.sr)
                    reverb_srcs_list.append(noise)
                    direct_srcs_list.append(noise)
                elif args.noise_type == 'white':
                    noise = np.random.randn(info_dict['mic_pos'].shape[0], clean_srcs_list_align[-1].shape[0])
                    reverb_srcs_list.append(noise)
                    direct_srcs_list.append(noise)
                elif args.noise_type == 'point':
                    reverb_noise, direct_noise = mch_rir_conv(clean_srcs_list_align[num_spk], rir[:, num_spk, :], early_rir_samples)
                    reverb_srcs_list.append(reverb_noise)
                    direct_srcs_list.append(direct_noise)
                elif args.noise_type == 'none':
                    noise = clean_srcs_list_align[-1]
                    noise = np.repeat(np.expand_dims(noise, 0), rir.shape[0], axis=0)
                    reverb_srcs_list.append(noise)
                    direct_srcs_list.append(noise)
                else:
                    raise ValueError("Noise type undefined.")

            reverb_srcs_list, gain_list = scale_audios(reverb_srcs_list, snr_list)
            direct_srcs_list = [direct_srcs_list[i] * np.power(10, (gain_list[i] / 20.)) for i in range(len(gain_list))]
            clean_srcs_list_align = [clean_srcs_list_align[i] * np.power(10, (gain_list[i] / 20.)) for i in range(len(gain_list))]

            reverb_srcs = np.stack(reverb_srcs_list, 0)
            direct_srcs = np.stack(direct_srcs_list, 0) 
            clean_srcs = np.stack(clean_srcs_list_align, 0)
            mixture = np.sum(reverb_srcs, 0)
        else:
            info_dict = {}
            info_dict['snr'] = np.array(snr_list)
            info_dict['start_sample'] = np.array(start_sample)
            info_dict['uttpath'] = utt_path_list
            clean_srcs, _ = scale_audios(clean_srcs, snr_list)
            mixture = np.sum(clean_srcs, 0)

        if args.normalize:
            max_sample = max(np.max(np.abs(mixture)), np.max(np.abs(clean_srcs)), np.max(np.abs(direct_srcs))) + 1e-12
            clean_srcs = clean_srcs / max_sample
            direct_srcs = direct_srcs / max_sample
            mixture = mixture / max_sample

        print('-' * 80)
        print("{:07d}".format(utt_idx))
        for k in info_dict.keys():
            if k == 'rir':
                print(k, info_dict[k].shape)
            else:
                print(k, info_dict[k])

        # Save audios and metadata
        fname = "{:07d}".format(utt_idx)
        output_path = '{}/wav/mix/{}.wav'.format(args.output_dir, fname)
        sf.write(output_path, np.transpose(mixture), args.sr)
        np.savez('{}/metadata/{}.npz'.format(args.output_dir, fname), **info_dict)
        wav_scp_file.write("{} {}\n".format(fname, output_path))
        reco2dur_file.write("{} {}\n".format(fname, duration))
        utt2spk_file.write("{} {}\n".format(fname, fname))

        assert num_spk + args.add_noise == len(clean_srcs)
        for i in range(len(clean_srcs)):
            if i == len(clean_srcs) - 1:
                if args.add_noise:
                    srcname = "noise"
                else:
                    srcname = "s{}".format(i+1)
            else:
                srcname = "s{}".format(i+1)
            sf.write('{}/wav/{}/{}.wav'.format(args.output_dir, srcname, fname), clean_srcs[i, :], args.sr)

        assert num_spk + args.add_noise == len(direct_srcs)
        for i in range(len(direct_srcs)):
            if i == len(direct_srcs) - 1:
                if args.add_noise:
                    srcname = "noise_direct"
                else:
                    srcname = "s{}_direct".format(i+1)
            else:
                srcname = "s{}_direct".format(i+1)
            sf.write('{}/wav/{}/{}.wav'.format(args.output_dir, srcname, fname), np.transpose(direct_srcs[i]), args.sr)

    wav_scp_file.close()
    reco2dur_file.close()
    utt2spk_file.close()
    return 0

if __name__ == '__main__':
    main()
