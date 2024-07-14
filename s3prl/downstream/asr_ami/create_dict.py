import os
import argparse

parser = argparse.ArgumentParser(description='Create dictionary')
parser.add_argument('text_file', type=str, help='text file')
parser.add_argument('output_file', type=str, help='output file')
args = parser.parse_args() 

def main():
    with open(args.text_file, 'r') as fh:
        content = fh.readlines()
    char_dict = {}
    for line in content:
        line = line.strip('\n')
        line_split = line.split(None, 1)
        text = line_split[1]
        text = " ".join(list(text.replace(" ", "|"))) + " |"
        char_list = text.split()
        for c in char_list:
            if c not in char_dict:
                char_dict[c] = 0
            char_dict[c] += 1
    char_dict_sorted = sorted(char_dict.items(), key=lambda x: -x[1])
    with open(args.output_file, 'w') as fh:
        for (char, cnt) in char_dict_sorted:
            fh.write("{} {}\n".format(char, cnt))
    return 0

if __name__ == '__main__':
    main()
