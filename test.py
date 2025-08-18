# test_random_image.py
# Minimal fast-reid feature test with a synthetic image

import argparse
import sys
import numpy as np
import torch.nn.functional as F

sys.path.append('.')  # so local fastreid imports work

from fast_reid.fastreid.config import get_cfg
from fast_reid.fastreid.utils.logger import setup_logger
from fast_reid.fastreid.utils.file_io import PathManager
from fast_reid.demo.predictor import FeatureExtractionDemo


def setup_cfg(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def main():
    parser = argparse.ArgumentParser("Minimal random-image feature extraction (fast-reid)")
    parser.add_argument(
        "--config-file", 
        required=False, 
        help="path to config.yml",
        default='/home/b08901172/.yuxiang/OinkTrack-reid-logs/config.yaml'
    )
    parser.add_argument(
        "--opts",
        default=[],
        nargs=argparse.REMAINDER,
        help="fast-reid KEY VALUE pairs, e.g. MODEL.WEIGHTS /path/to/weights.pth",
    )
    parser.add_argument("--height", type=int, default=None, help="override input height")
    parser.add_argument("--width", type=int, default=None, help="override input width")
    parser.add_argument("--seed", type=int, default=42, help="rng seed for reproducibility")
    parser.add_argument("--output", default="feat.npy", help="where to save the feature")
    args = parser.parse_args()

    setup_logger(name="fastreid")

    cfg = setup_cfg(args)
    H = args.height if args.height is not None else int(cfg.INPUT.SIZE_TEST[0])
    W = args.width  if args.width  is not None else int(cfg.INPUT.SIZE_TEST[1])

    # Make a synthetic uint8 image in HxWx3 (like cv2.imread would)
    rng = np.random.default_rng(args.seed)
    img = rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)
    print(img.shape)

    demo = FeatureExtractionDemo(cfg, parallel=False)

    # Forward pass -> normalize -> numpy
    feat = demo.run_on_image(img)           # torch.Tensor [1, C] or [C]
    feat = F.normalize(feat)                # cosine-normalized
    feat = feat.detach().cpu().numpy()      # to numpy

    # Save + print some quick info
    PathManager.mkdirs(".")

    print(f"Synthetic image shape: ({H}, {W}, 3), dtype=uint8")
    print(f"Feature shape: {feat.shape}, dtype={feat.dtype}")
    print(f"First 8 values: {np.round(feat.reshape(-1)[:8], 4)}")

if __name__ == "__main__":
    main()

