"""Verify a downloaded NavisSteer or LinearSteer checkpoint locally."""

import argparse
from pathlib import Path

import numpy as np

from navissteer_model import load_steering_runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default="models/navissteer_best.pt",
        help="path to the checkpoint downloaded from Google Drive",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    runtime = load_steering_runtime(checkpoint_path)

    # A dummy frame validates architecture, preprocessing, and inference.
    dummy_rgb = np.zeros((720, 1280, 3), dtype=np.uint8)
    steering = runtime.predict(dummy_rgb)

    print("Checkpoint:", checkpoint_path.resolve())
    print("Model:", runtime.model_name)
    print("Device:", runtime.device)
    print("Input mode:", runtime.input_mode)
    print("Image size:", runtime.image_size)
    if runtime.input_mode == "roi":
        print("Source image size:", runtime.source_image_size)
        print("ROI crop (left, top, right, bottom):", runtime.roi_crop)
    print("Epoch:", runtime.metadata.get("epoch", "not recorded"))
    print("Validation MSE:", runtime.metadata.get("validation_mse", "not recorded"))
    print("Dummy-frame steering:", steering)
    print("CHECKPOINT READY")


if __name__ == "__main__":
    main()
