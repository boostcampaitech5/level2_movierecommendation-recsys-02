import sys
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import argparse
import os
import glob

def filepaths(args):
    filepaths = glob.glob('*.csv')
    if args.pick is not None:
        filepaths = [i for i in filepaths if args.pick in i]
    print(filepaths)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pick", default=None, type=str)
    parser.add_argument("--ratio", default=None, type=list)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    filepaths(args)
