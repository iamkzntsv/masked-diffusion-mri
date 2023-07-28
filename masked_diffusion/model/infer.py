import argparse

from ..etl.slice_extractor import SliceExtractor
from ..utils import load_mri


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=int, help="Path to a nifti image")
    return parser.parse_args()


def main():
    args = parse_args()
    slice_ext = SliceExtractor()
    t1 = load_mri("")
    print(args)


if __name__ == "__main__":
    main()
