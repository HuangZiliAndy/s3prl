import os
import time
import argparse
import subprocess

parser = argparse.ArgumentParser(description='Beamformit')
parser.add_argument('wav_scp_file', type=str, help='wav scp file')
parser.add_argument('output_dir', type=str, help='output directory')
parser.add_argument('channels', type=str, help='channels')
parser.add_argument('--config_file', type=str, help='config file')
parser.add_argument('--split', type=int, default=0, help='split number')
args = parser.parse_args()

def main():
    if not os.path.exists(args.output_dir + '/tmp_{}'.format(args.split)):
        os.makedirs(args.output_dir + '/tmp_{}'.format(args.split))
    channels = [int(c) for c in args.channels.split(',')]
    with open(args.wav_scp_file, 'r') as fh:
        content = fh.readlines()
    cnt = 0
    start_time = time.time()
    for line in content:
        line = line.strip('\n')
        line_split = line.split()
        uttname, wav_path = line_split[0], line_split[1]
        channel_str = "{}".format(uttname)
        assert os.path.exists(wav_path)
        # temporary store selected channels
        for c in channels:
            cmd = "sox {} {}/tmp_{}/{}_{}.wav remix {}".format(wav_path, args.output_dir, args.split, uttname, c, c+1)
            status, output = subprocess.getstatusoutput(cmd)
            assert status == 0
            channel_str += " " + "{}_{}.wav".format(uttname, c) 
        channel_str += '\n'
        channel_file = "{}/tmp_{}/channels".format(args.output_dir, args.split)
        with open(channel_file, 'w') as fh:
            fh.write(channel_str)
        cmd = "BeamformIt -s {} -c {} --config_file {} --source_dir {}/tmp_{} --result_dir {}".format(uttname, channel_file, args.config_file, args.output_dir, args.split, args.output_dir)
        status, output = subprocess.getstatusoutput(cmd)
        assert status == 0
        # remove extra files
        for c in channels:
            os.remove("{}/tmp_{}/{}_{}.wav".format(args.output_dir, args.split, uttname, c))
        for ext in ["del", "del2", "info", "weat"]:
            os.remove("{}/{}.{}".format(args.output_dir, uttname, ext))
        cnt += 1
        print("{}/{}".format(cnt, len(content)), flush=True)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.6f} seconds")
    return 0

if __name__ == '__main__':
    main()
