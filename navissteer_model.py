"""NavisSteer architecture and checkpoint-loading utilities.

Keep this architecture synchronized with the NavisSteer class in NavisAI.ipynb.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NavisSteer(nn.Module):
    """CNN regression model that predicts one CARLA steering value."""

    def __init__(self, input_channels=1):
        super().__init__()

        self.input_channels = int(input_channels)
        if self.input_channels not in (1, 3):
            raise ValueError(
                f"NavisSteer supports 1 or 3 input channels, got {input_channels}."
            )

        self.conv1 = nn.Conv2d(self.input_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))

        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, images):
        images = self.pool(F.relu(self.conv1(images)))
        images = self.dropout(images)

        images = self.pool(F.relu(self.conv2(images)))
        images = self.dropout(images)

        images = self.pool(F.relu(self.conv3(images)))
        images = self.dropout(images)

        images = self.global_pool(images)
        images = images.view(images.size(0), -1)

        images = F.relu(self.fc1(images))
        images = self.dropout(images)
        return self.fc2(images)


def load_navissteer_checkpoint(checkpoint_path, device=None):
    """Load a NavisSteer checkpoint and return (model, metadata, device)."""

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"NavisSteer checkpoint not found: {checkpoint_path}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(
            "Checkpoint must be a dictionary containing 'model_state_dict'."
        )
    if checkpoint.get("model_name", "NavisSteer") != "NavisSteer":
        raise ValueError(
            f"Expected a NavisSteer checkpoint, got {checkpoint.get('model_name')!r}."
        )

    state_dict = checkpoint["model_state_dict"]
    input_channels = int(
        checkpoint.get("input_channels", state_dict["conv1.weight"].shape[1])
    )

    model = NavisSteer(input_channels=input_channels).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, checkpoint, device


class NavisSteerRuntime:
    """Preprocess CARLA RGB frames and run NavisSteer inference."""

    def __init__(self, checkpoint_path, device=None):
        self.model, self.metadata, self.device = load_navissteer_checkpoint(
            checkpoint_path,
            device=device,
        )
        self.image_size = tuple(self.metadata.get("image_size", (220, 220)))
        if self.image_size != (220, 220):
            raise ValueError(
                f"This runtime expects a 220x220 checkpoint, got {self.image_size}."
            )
        self.input_channels = int(
            self.metadata.get(
                "input_channels",
                self.model.conv1.weight.shape[1],
            )
        )

    def preprocess_rgb(self, rgb_array):
        """Convert an HxWx3 uint8 RGB array to checkpoint-compatible input."""

        rgb_array = np.asarray(rgb_array)
        if rgb_array.ndim != 3 or rgb_array.shape[2] != 3:
            raise ValueError(f"Expected an HxWx3 RGB image, got {rgb_array.shape}")

        rgb_array = np.ascontiguousarray(rgb_array)
        image = torch.from_numpy(rgb_array).to(self.device, non_blocking=True)
        image = image.permute(2, 0, 1).unsqueeze(0).float().div_(255.0)

        if self.input_channels == 1:
            # Match the old notebook's grayscale conversion before resizing.
            red, green, blue = image[:, 0:1], image[:, 1:2], image[:, 2:3]
            image = 0.299 * red + 0.587 * green + 0.114 * blue

        image = F.interpolate(
            image,
            size=self.image_size,
            mode="bilinear",
            align_corners=False,
        )
        return image

    def predict(self, rgb_array):
        """Return a CARLA-compatible steering value clamped to [-1, 1]."""

        image = self.preprocess_rgb(rgb_array)
        with torch.inference_mode():
            steering = float(self.model(image).squeeze().item())
        return max(-1.0, min(1.0, steering))
