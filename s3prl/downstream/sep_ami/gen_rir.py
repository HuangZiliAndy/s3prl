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

parser = argparse.ArgumentParser(description='Generate multi-channel audios')
parser.add_argument('output_dir', type=str, help='output directory')
parser.add_argument('--mic_arch', type=str, default='AMI', help='microphone architectures')
parser.add_argument('--max_room', type=str, default='9,9,4', help='max room x,y,z')
parser.add_argument('--min_room', type=str, default='3,3,2.5', help='min room x,y,z')
parser.add_argument('--sr', type=int, default=16000, help='sample rate')
parser.add_argument('--rt60', type=str, default='0.05,0.6', help='range of rt60')
parser.add_argument('--num_rirs', type=int, default=10, help='number of rirs to generate')
parser.add_argument('--num_srcs', type=int, default=3, help='number of sources')
parser.add_argument('--seed', type=int, default=7, help='random seed')
args = parser.parse_args()

def sample_room_dim(max_room, min_room):
    return np.random.uniform(np.array(max_room), np.array(min_room))

def sample_src_pos(room_dim, num_src, array_pos,
                   min_mic_dis, max_mic_dis, min_dis_wall):
    # random sample the source positon,
    src_pos = []
    while(len(src_pos) < num_src):
        pos = np.random.uniform(np.array(min_dis_wall), np.array(
            room_dim) - np.array(min_dis_wall))
        dis = np.linalg.norm(pos - np.array(array_pos))
        
        if dis >= min_mic_dis and dis <= max_mic_dis:
            src_pos.append(pos)
    return src_pos

def generate_mic_array_pos(mic_arch, array_pos, room_dim, min_dis_wall=[0.5, 0.5, 0.5]):
    """
    Generate the microphone array position according to the given microphone architecture (geometry)
    :param mic_arch: np.array with shape [n_mic, 3]
                    the relative 3D coordinate to the array_pos in (m)
                    e.g., 2-mic LA (left->right) [[-0.1, 0, 0], [0.1, 0, 0]];
                    e.g., 4-mic CA (north->clockwise) [[0, 0.035, 0], [0.035, 0, 0], [0, -0.035, 0], [-0.035, 0, 0]]
    :param array_pos: array CENTER position in (m)
    :param room_dim: room dimension in (m)
    :param min_dis_wall: minimum distance from the wall in (m)
    :return
        mic_pos: microphone array position in (m) with shape [n_mic, 3]
        array_pos: array CENTER position in (m) with shape [1, 3]
    """
    mic_array_center = np.mean(mic_arch, 0, keepdims=True)  # [1, 3]
    max_radius = max(np.linalg.norm(mic_arch - mic_array_center, axis=-1))
    array_pos = np.random.uniform(np.array(min_dis_wall) + max_radius,
                                  np.array(room_dim) - np.array(min_dis_wall) - max_radius).reshape(1, 3)

    mic_pos = array_pos + mic_arch  # [n_mic, 3]
    return mic_pos, array_pos

def get_location_info(array_pos, src_pos):
    azm_rads, ele_rads, dists, min_ad = get_precise_azm_ele_dist(
        array_pos, np.array(src_pos))

    info_dict = {
        'azm_rad': azm_rads,
        'azm_deg': azm_rads/math.pi*180.0,
        'ele_rad': ele_rads,
        'ele_deg': ele_rads/math.pi*180.0,
        'dists': dists,
        'min_ad': min_ad,
        'src_pos': src_pos,
        'array_pos': array_pos,
    }
    # info_str = "azimuths: {}\nelevations: {}\ndistances: {}\nmin_ad: {:.2f}".format(
    # azm_rads/math.pi*180.0, ele_rads/math.pi*180.0, dists, min_ad)
    # print(info_str)
    return info_dict

def get_precise_azm_ele_dist(array_pos, src_pos):
    """
    Get precise azimuth [-pi, pi], elevation [0, pi] and src2array distance according to given array center position and source position.
    :param array_pos: np.array with shape [1, 3]
    :param src_pos: np.array with shape [n_source, 3]
    :return
        azm_rads: azimuths in rad [n_source]
        ele_rads: elevations in rad [n_source]
        dists: distances in (m) [n_source]
        min_ad: scalar, the minimum angle difference between multiple sources
    """

    azm_rads = []
    azm_degrees = []
    ele_rads = []
    dists = []
    array_pos = array_pos.reshape(-1)
    for src_idx in range(src_pos.shape[0]):
        src = src_pos[src_idx]
        # calculate the distance
        xyz_distance = np.linalg.norm(array_pos - src)
        xy_distance = np.linalg.norm(array_pos[:2] - src[:2])
        # calculate the azimuth (x-y coordinate)
        azm_rad = math.atan2(
            src[1] - array_pos[1], src[0] - array_pos[0])
        azm_degree = azm_rad * (180.0 / math.pi)

        # calculate the elevation (xy-z coordinate)
        ele_rad = math.atan2(src[2] - array_pos[2], xy_distance)

        dists.append(xyz_distance)
        azm_rads.append(azm_rad)
        ele_rads.append(abs(ele_rad))
        azm_degrees.append(azm_degree)

    min_ad = get_min_angle_diff(azm_degrees)
    return np.array(azm_rads), np.array(ele_rads), np.array(dists), min_ad

def angle_diff(a, b):
    """in degrees"""
    return min(abs(a - b), 360 - abs(a - b))

def get_min_angle_diff(azm_degrees):
    """
    Get the minimum angle difference between multiple sources
    :param azm_degrees: [n_source]
    :return 
        min_ad: scalar, the minimum angle difference between multiple sources, 
    """
    min_ad = 181
    if len(azm_degrees) <= 1:
        return -1
    for i in range(len(azm_degrees)):
        for j in range(len(azm_degrees)):
            if i == j:
                continue
            ad = angle_diff(azm_degrees[j], azm_degrees[i])
            if ad <= min_ad:
                min_ad = ad
    return min_ad

def PRA_RIR_MC(mic_arch, sr, rt60, room_dim, array_pos, src_pos, num_src):
    """
    Generate RIRs for the given array using pyroomacoustic toolkit.
    :param mic_arch: microphone architecture with shape [n_mic, 3]
    :return
        rir: generated RIRs with shape [n_mic, n_src, rir_len]
        rir_direct: generated [anechoic] RIRs  
    """
    if len(rt60) == 2:
        rt60 = np.random.uniform(rt60[0], rt60[1])
    else:
        rt60 = rt60[0]

    # We invert Sabine's formula to obtain the parameters for the ISM simulator
    while True:
        try:
            e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)
            break
        except:
            rt60 = rt60 + 0.1

    # Def room
    room = pra.ShoeBox(
        room_dim, fs=sr, materials=pra.Material(e_absorption), max_order=max_order
    )

    # Mic position
    mic_arch = np.array(mic_arch)
    num_mic = mic_arch.shape[0]
    
    if array_pos is None:
        mic_pos, array_pos = generate_mic_array_pos(
            mic_arch, array_pos, room_dim=room_dim)
    else:
        mic_pos = array_pos + mic_arch
    room.add_microphone_array(pra.MicrophoneArray(mic_pos.T, room.fs))

    # Source position
    if src_pos is None:
        src_pos = sample_src_pos(room_dim=room_dim, num_src=num_src, array_pos=array_pos, min_mic_dis=0.2, max_mic_dis=5, min_dis_wall=[0.5, 0.5, 0.5])
        assert len(src_pos) == num_src
    else:
        assert len(src_pos) == num_src

    for pos in src_pos:
        room.add_source(pos, signal=None, delay=0.0)

    # Compute RIR
    room.compute_rir()

    # Pad rir
    # import pdb; pdb.set_trace()
    rir = np.asarray(room.rir, dtype=object)  # , dtype=np.float32)
    if rir.ndim != 3:
        rir = rir.reshape(-1)
        max_len = max([rir[i].shape[-1] for i in range(rir.shape[-1])])
        for i in range(num_mic):
            for j in range(num_src):
                room.rir[i][j] = np.pad(room.rir[i][j],
                                        (0, max_len - room.rir[i][j].shape[-1]), 'constant', constant_values=(0, 0))
    rir = np.asarray(room.rir, dtype=np.float32)

    info_dict = get_location_info(array_pos, np.array(src_pos))
    info_dict["room_size"] = room_dim
    info_dict["rt60"] = rt60
    info_dict["mic_pos"] = mic_pos

    return rir, None, info_dict

def main():
    print(args)

    np.random.seed(args.seed)

    max_room= [float(s) for s in args.max_room.split(',')]
    min_room= [float(s) for s in args.min_room.split(',')]
    rt60 = [float(s) for s in args.rt60.split(',')] 
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.mic_arch == "AMI":
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        mic_arch = [[0.1 * np.cos(angle), 0.1 * np.sin(angle), 0] for angle in angles]
    else:
        raise ValueError("Condition not defined.")
    mic_arch = np.round(np.array(mic_arch), decimals=10)

    for i in range(args.num_rirs):
        room_dim = sample_room_dim(max_room=max_room, min_room=min_room)
        rir, rir_direct, info_dict = PRA_RIR_MC(mic_arch, sr=args.sr, rt60=rt60, room_dim=room_dim, array_pos=None, src_pos=None, num_src=args.num_srcs)
        info_dict['rir'] = rir
        print("-" * 80)
        print("{}/{}".format(i+1, args.num_rirs))
        for k in info_dict.keys():
            if k == 'rir':
                print(k, info_dict[k].shape)
            else:
                print(k, info_dict[k])

        np.savez('{}/{:07d}.npz'.format(args.output_dir, i+1), **info_dict)
    return 0

if __name__ == '__main__':
    main()
