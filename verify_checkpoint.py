"""Verify that a downloaded NavisSteer checkpoint loads on this computer."""

import argparse
from pathlib import Path

import numpy as np

from navissteer_model import NavisSteerRuntime


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
    runtime = NavisSteerRuntime(checkpoint_path)

    # A dummy frame validates architecture, preprocessing, and inference.
    dummy_rgb = np.zeros((720, 1280, 3), dtype=np.uint8)
    steering = runtime.predict(dummy_rgb)

    print("Checkpoint:", checkpoint_path.resolve())
    print("Device:", runtime.device)
    print("Image size:", runtime.image_size)
    print("Epoch:", runtime.metadata.get("epoch", "not recorded"))
    print("Validation MSE:", runtime.metadata.get("validation_mse", "not recorded"))
    print("Dummy-frame steering:", steering)
    print("CHECKPOINT READY")


if __name__ == "__main__":
    main()
