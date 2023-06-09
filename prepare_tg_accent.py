import argparse
import glob
from os.path import expanduser, join
import pathlib
from pathlib import Path
import re
import sys

from tqdm import tqdm

from convert_label import read_lab


# full context label to accent label from ttslearn
def numeric_feature_by_regex(regex, s):
    match = re.search(regex, s)
    if match is None:
        return -50
    return int(match.group(1))


def pp_symbols(labels, drop_unvoiced_vowels=True):
    PP = []
    N = len(labels)

    for n in range(len(labels)):
        lab_curr = labels[n]


        p3 = re.search(r"\-(.*?)\+", lab_curr).group(1)

        if drop_unvoiced_vowels and p3 in "AEIOU":
            p3 = p3.lower()

        if p3 == 'sil':
            assert n== 0 or n == N-1
            if n == N-1:
                e3 = numeric_feature_by_regex(r"!(\d+)_", lab_curr)
                if e3 == 0:
                    PP.append("")
                elif e3 == 1:
                    PP.append("")
            continue
        elif p3 == "pau":
            PP.append("sp")
            continue
        else:
            PP.append(p3)
    return PP


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('lab',type=str,help='path to lab files. this program searchs for .lab files in specified directory and subdirectories')
    parser.add_argument('output',type=str,help='path to output accent and tg file')
    parser.add_argument('speaker',type=str,help='speaker name')
    parser.add_argument('--with_accent', type=bool,help='whether you want accent labels or not')

    args = parser.parse_args()
    # lab_files = pathlib.Path(args.lab).glob('**/*.lab')
    lab_files = glob.glob(join(args.lab, '*.lab'))

    # create output directory
    
    tg_dir = (Path(args.output) / 'TextGrid'/args.speaker)
    ac_dir = (Path(args.output)/ 'accent')
    if not tg_dir.exists():
        tg_dir.mkdir(parents=True)
    if not ac_dir.exists():
        ac_dir.mkdir()

    # iter through lab files
    for lab_file in tqdm(lab_files):
        lab_file = pathlib.Path(lab_file)
        if args.with_accent:
            accent = []
            with open(lab_file) as f:
                lines = f.readlines()
            lab, accent = pp_symbols(lines)
            with open(ac_dir/ lab_file.with_suffix('.accent').name,mode='w') as f:
                f.writelines([''.join(accent)])
        
        label = read_lab(str(lab_file))
        textgridFilePath = tg_dir/lab_file.with_suffix('.TextGrid').name
        label.to_textgrid(textgridFilePath)

        

    
