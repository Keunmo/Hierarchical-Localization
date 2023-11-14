import os
from pathlib import Path
import re

def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


def reindexing(data: Path):
    idx = 0
    files = os.listdir(data)
    files.sort(key=natural_keys)
    for file in files:
        os.rename(data / file, data / f'{idx:d}.jpg')
        idx += 1


if __name__ == '__main__':
    data = Path("/home/keunmo/workspace/dataset/s20fe-robjet-uw-43/images")
    reindexing(data)