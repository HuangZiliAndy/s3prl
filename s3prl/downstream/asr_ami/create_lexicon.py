import os
import argparse

parser = argparse.ArgumentParser(description='Create lexicon file')
parser.add_argument('input_file', type=str, help='Input text file')
parser.add_argument('output_file', type=str, help='Output lexicon file')
args = parser.parse_args()

def main():
    word_dict = {}
    with open(args.input_file, 'r') as fh:
        content = fh.readlines()
    for line in content:
        line = line.strip('\n')
        line_split = line.split()
        for word in line_split:
            if word not in word_dict:
                word_dict[word] = 0
            word_dict[word] += 1
    word_list = list(word_dict.keys())
    word_list.sort()
    print("{} words in total".format(len(word_list)))
    with open(args.output_file, 'w') as fh:
        for word in word_list:
            fh.write("{}\t{}\n".format(word, ' '.join(list(word+'|'))))
    return 0

if __name__ == '__main__':
    main()
