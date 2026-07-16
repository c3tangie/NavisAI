"""NavisSteer architecture and checkpoint-loading utilities.

Keep this architecture synchronized with the NavisSteer class in NavisAI.ipynb.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# The ROI dataset contains the bottom half of each original 220x220 RGB frame.
# Tensor/image sizes use (height, width); crop coordinates use
# (left, top, right, bottom).
FULL_FRAME_SIZE = (220, 220)
ROI_INPUT_SIZE = (110, 220)
BOTTOM_HALF_ROI = (0, 110, 220, 220)


class NavisSteer(nn.Module):
    """CNN regression model for RGB ROI tensors shaped [N, 3, 110, 220]."""

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


class LinearSteer(nn.Module):
    """Flattened-RGB linear-regression steering baseline."""

    def __init__(self, width=220, height=110, input_channels=3):
        super().__init__()
        self.width = int(width)
        self.height = int(height)
        self.input_channels = int(input_channels)
        self.flatten = nn.Flatten(start_dim=1)
        self.linear = nn.Linear(
            self.input_channels * self.width * self.height,
            1,
        )

    def forward(self, images):
        flattened_images = self.flatten(images)
        expected_features = self.input_channels * self.width * self.height
        if flattened_images.shape[1] != expected_features:
            raise ValueError(
                f"LinearSteer expected {expected_features} input values, got "
                f"{flattened_images.shape[1]}."
            )
        return self.linear(flattened_images)


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


def load_linearsteer_checkpoint(checkpoint_path, device=None):
    """Load a LinearSteer checkpoint and return (model, metadata, device)."""

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"LinearSteer checkpoint not found: {checkpoint_path}")

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
    if checkpoint.get("model_name", "LinearSteer") != "LinearSteer":
        raise ValueError(
            f"Expected a LinearSteer checkpoint, got "
            f"{checkpoint.get('model_name')!r}."
        )

    state_dict = checkpoint["model_state_dict"]
    if "linear.weight" not in state_dict:
        raise ValueError("LinearSteer checkpoint is missing 'linear.weight'.")

    image_size = tuple(
        int(value) for value in checkpoint.get("image_size", ROI_INPUT_SIZE)
    )
    if len(image_size) != 2 or min(image_size) <= 0:
        raise ValueError(f"Invalid checkpoint image_size metadata: {image_size}.")
    height, width = image_size
    flattened_features = int(state_dict["linear.weight"].shape[1])
    pixels = height * width
    if flattened_features % pixels != 0:
        raise ValueError(
            f"LinearSteer has {flattened_features} features, which is incompatible "
            f"with image_size={image_size}."
        )
    inferred_channels = flattened_features // pixels
    input_channels = int(checkpoint.get("input_channels", inferred_channels))
    if input_channels != inferred_channels:
        raise ValueError(
            f"Checkpoint metadata says {input_channels} channels, but its linear "
            f"weights require {inferred_channels}."
        )

    model = LinearSteer(
        width=width,
        height=height,
        input_channels=input_channels,
    ).to(device)
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
        self.model_name = "NavisSteer"
        self._configure_preprocessing()

    def _configure_preprocessing(self):
        self.image_size = tuple(
            int(value) for value in self.metadata.get("image_size", FULL_FRAME_SIZE)
        )
        if len(self.image_size) != 2 or min(self.image_size) <= 0:
            raise ValueError(
                f"Invalid checkpoint image_size metadata: {self.image_size}."
            )

        default_input_mode = "roi" if self.image_size == ROI_INPUT_SIZE else "full"
        self.input_mode = str(
            self.metadata.get("input_mode", default_input_mode)
        ).lower()
        if self.input_mode not in ("full", "roi"):
            raise ValueError(
                f"Unsupported NavisSteer input mode: {self.input_mode!r}."
            )

        self.source_image_size = tuple(
            int(value)
            for value in self.metadata.get("source_image_size", FULL_FRAME_SIZE)
        )
        if len(self.source_image_size) != 2 or min(self.source_image_size) <= 0:
            raise ValueError(
                "Invalid checkpoint source_image_size metadata: "
                f"{self.source_image_size}."
            )

        raw_roi = self.metadata.get("roi_crop", BOTTOM_HALF_ROI)
        self.roi_crop = (
            tuple(int(value) for value in raw_roi)
            if raw_roi is not None
            else None
        )
        if self.input_mode == "roi":
            if self.source_image_size != FULL_FRAME_SIZE:
                raise ValueError(
                    "ROI NavisSteer expects source_image_size=(220, 220), got "
                    f"{self.source_image_size}."
                )
            if self.image_size != ROI_INPUT_SIZE:
                raise ValueError(
                    "ROI NavisSteer expects image_size=(110, 220), got "
                    f"{self.image_size}."
                )
            if self.roi_crop is None or len(self.roi_crop) != 4:
                raise ValueError("ROI checkpoints require four roi_crop coordinates.")
            if self.roi_crop != BOTTOM_HALF_ROI:
                raise ValueError(
                    "ROI NavisSteer must discard the top half using "
                    f"roi_crop={BOTTOM_HALF_ROI}, got {self.roi_crop}."
                )
            left, top, right, bottom = self.roi_crop
            source_height, source_width = self.source_image_size
            if not (
                0 <= left < right <= source_width
                and 0 <= top < bottom <= source_height
            ):
                raise ValueError(
                    f"Invalid roi_crop {self.roi_crop} for source image size "
                    f"{self.source_image_size}."
                )
        self.input_channels = int(
            self.metadata.get(
                "input_channels",
                self.model.input_channels,
            )
        )

    def preprocess_rgb(self, rgb_array):
        """Resize to 220x220, discard its top half, and return 3x110x220."""

        rgb_array = np.asarray(rgb_array)
        if rgb_array.ndim != 3 or rgb_array.shape[2] != 3:
            raise ValueError(f"Expected an HxWx3 RGB image, got {rgb_array.shape}")

        rgb_array = np.ascontiguousarray(rgb_array)
        image = torch.from_numpy(rgb_array).to(self.device, non_blocking=True)
        image = image.permute(2, 0, 1).unsqueeze(0).float().div_(255.0)

        if self.input_mode == "roi":
            # Match training exactly: first normalize the CARLA sensor geometry
            # to 220x220, then discard rows 0:110 and retain rows 110:220.
            image = F.interpolate(
                image,
                size=self.source_image_size,
                mode="bilinear",
                align_corners=False,
            )
            left, top, right, bottom = self.roi_crop
            image = image[:, :, top:bottom, left:right]

        if image.shape[-2:] != self.image_size:
            image = F.interpolate(
                image,
                size=self.image_size,
                mode="bilinear",
                align_corners=False,
            )

        if self.input_channels == 1:
            # Retain compatibility with older grayscale checkpoints.
            red, green, blue = image[:, 0:1], image[:, 1:2], image[:, 2:3]
            image = 0.299 * red + 0.587 * green + 0.114 * blue
        return image

    def predict(self, rgb_array):
        """Return a CARLA-compatible steering value clamped to [-1, 1]."""

        image = self.preprocess_rgb(rgb_array)
        with torch.inference_mode():
            steering = float(self.model(image).squeeze().item())
        return max(-1.0, min(1.0, steering))


class LinearSteerRuntime(NavisSteerRuntime):
    """Preprocess CARLA RGB frames and run the LinearSteer baseline."""

    def __init__(self, checkpoint_path, device=None):
        self.model, self.metadata, self.device = load_linearsteer_checkpoint(
            checkpoint_path,
            device=device,
        )
        self.model_name = "LinearSteer"
        self._configure_preprocessing()


def load_steering_runtime(checkpoint_path, device=None):
    """Auto-detect and load a NavisSteer or LinearSteer checkpoint."""

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model_name = checkpoint.get("model_name") if isinstance(checkpoint, dict) else None
    if model_name == "NavisSteer":
        return NavisSteerRuntime(checkpoint_path, device=device)
    if model_name == "LinearSteer":
        return LinearSteerRuntime(checkpoint_path, device=device)
    raise ValueError(
        "Checkpoint model_name must be 'NavisSteer' or 'LinearSteer', got "
        f"{model_name!r}."
    )
