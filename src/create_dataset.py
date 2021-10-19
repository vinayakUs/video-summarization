import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-dir', type=str, required=True, help='video directory location ')
    parser.add_argument('--sample-rate', type=int, default=15, help='sample rate of video ')
    parser.add_argument('--save-path', type=str, default='custom_data/custom_dataset.h5',
                        help='save path of dataset.h5')
    parser.add_argument('--summary-dir', type=str, required=True, help='video directory location ')
    args = parser.parse_args()

    # create output directory
    out_dir = Path(args.save_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # feature extractor
    print('Loading feature extractor ...')
    pass
