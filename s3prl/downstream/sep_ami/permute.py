import os
import argparse
from jiwer import wer
import editdistance

parser = argparse.ArgumentParser(description='Prepare ref_text and pred_text for scoring')
parser.add_argument('ref_text', type=str, help='reference text file')
parser.add_argument('pred_text', type=str, help='predict text file')
parser.add_argument('output_dir', type=str, help='output directory')
parser.add_argument('--token', type=str, default='word', help='token level')
args = parser.parse_args()

def main():
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(args.ref_text, 'r') as fh:
        content_ref = fh.readlines()
    with open(args.pred_text, 'r') as fh:
        content_pred = fh.readlines()
    utt2ref_text = {}
    for line in content_ref:
        line = line.strip('\n')
        line_split = line.split(None, 1)
        utt, text = line_split[0], line_split[1]
        text_per_spk = text.split('#')
        text_list = [" ".join(text.split()[1:]) for text in text_per_spk]
        text_list = [text for text in text_list if text != ""]
        utt2ref_text[utt] = text_list
    utt2pred_text = {}
    for line in content_pred:
        line = line.strip('\n')
        line_split = line.split(None, 1)
        if len(line_split) == 2:
            utt, text = line_split[0], line_split[1]
        else:
            utt, text = line_split[0], ""
        utt = '_'.join(utt.split('_')[:-1])
        if utt not in utt2pred_text:
            utt2pred_text[utt] = []
        utt2pred_text[utt].append(text)
    assert len(utt2ref_text) == len(utt2pred_text)
    
    if len(list(utt2ref_text.values())[0]) == 2:
        utt2pred_text_new = {}
        for utt in utt2ref_text.keys():
            ref_text_list = utt2ref_text[utt]
            pred_text_list = utt2pred_text[utt]
            assert len(ref_text_list) == len(pred_text_list) == 2
            if args.token == 'word':
                error1 = editdistance.eval(pred_text_list[0].split(), ref_text_list[0].split()) + editdistance.eval(pred_text_list[1].split(), ref_text_list[1].split())
                error2 = editdistance.eval(pred_text_list[0].split(), ref_text_list[1].split()) + editdistance.eval(pred_text_list[1].split(), ref_text_list[0].split())
            elif args.token == 'char':
                error1 = editdistance.eval(list(pred_text_list[0]), list(ref_text_list[0])) + editdistance.eval(list(pred_text_list[1]), list(ref_text_list[1]))
                error2 = editdistance.eval(list(pred_text_list[0]), list(ref_text_list[1])) + editdistance.eval(list(pred_text_list[1]), list(ref_text_list[0]))
            else:
                raise NotImplementedError

            if error1 <= error2:
                utt2pred_text_new[utt] = pred_text_list
            else:
                utt2pred_text_new[utt] = pred_text_list[::-1]
        utt2pred_text = utt2pred_text_new

    uttlist = list(utt2ref_text.keys())
    uttlist.sort()
    ref_text_file = open("{}/ref.ark".format(args.output_dir), 'w')
    hyp_text_file = open("{}/hyp.ark".format(args.output_dir), 'w')
    utt2spk_file = open("{}/utt2spk".format(args.output_dir), 'w')
    for utt in uttlist:
        assert len(utt2ref_text[utt]) == len(utt2pred_text[utt])
        assert len(utt2ref_text[utt]) == 1 or len(utt2ref_text[utt]) == 2
        for i in range(len(utt2ref_text[utt])):
            ref_text_file.write("{}_s{} {}\n".format(utt, i+1, utt2ref_text[utt][i]))
            hyp_text_file.write("{}_s{} {}\n".format(utt, i+1, utt2pred_text[utt][i]))
            utt2spk_file.write("{}_s{} {}_s{}\n".format(utt, i+1, utt, i+1))
    ref_text_file.close()
    hyp_text_file.close()
    utt2spk_file.close()
    return 0

if __name__ == '__main__':
    main()
