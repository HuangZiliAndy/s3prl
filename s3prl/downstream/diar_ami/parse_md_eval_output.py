import os
import re
import sys

def main():
    input_string = sys.stdin.read()
    pattern = r'OVERALL SPEAKER DIARIZATION ERROR = (\d+\.\d+) percent of scored speaker time'
    match = re.search(pattern, input_string)
    assert match
    DER = float(match.group(1))

    pattern = r'MISSED SPEAKER TIME =\s+(\d+\.\d+) secs \(\s+(\d+\.\d+) percent of scored speaker time\)'
    match = re.search(pattern, input_string)
    assert match
    MISS = float(match.group(2))

    pattern = r'FALARM SPEAKER TIME =\s+(\d+\.\d+) secs \(\s+(\d+\.\d+) percent of scored speaker time\)'
    match = re.search(pattern, input_string)
    assert match
    FA = float(match.group(2))

    pattern = r'SPEAKER ERROR TIME =\s+(\d+\.\d+) secs \(\s+(\d+\.\d+) percent of scored speaker time\)'
    match = re.search(pattern, input_string)
    assert match
    CF = float(match.group(2))

    print("{:.2f} {:.2f} {:.2f} {:.2f}".format(DER, MISS, FA, CF))
    return 0

if __name__ == '__main__':
    main()
