"""NavisVision architecture, checkpoint loading, and CARLA-frame inference."""

from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


CLASS_NAMES = (
    "bike",
    "motobike",
    "person",
    "traffic_light_green",
    "traffic_light_orange",
    "traffic_light_red",
    "traffic_sign_30",
    "traffic_sign_60",
    "traffic_sign_90",
    "vehicle",
)

TRAFFIC_CONTROL_CLASS_NAMES = (
    "traffic_light_green",
    "traffic_light_orange",
    "traffic_light_red",
    "traffic_sign_30",
    "traffic_sign_60",
    "traffic_sign_90",
)

FINE_TO_DETECTION_GROUP = {
    "bike": "bike",
    "motobike": "motobike",
    "person": "person",
    "traffic_light_green": "traffic_light",
    "traffic_light_orange": "traffic_light",
    "traffic_light_red": "traffic_light",
    "traffic_sign_30": "speed_sign",
    "traffic_sign_60": "speed_sign",
    "traffic_sign_90": "speed_sign",
    "vehicle": "vehicle",
}

V3_DEFAULT_GRID_SIZES = (52, 26, 13)


class ConvBlock(nn.Module):
    """Convolution, normalization, activation, and 2x downsampling."""

    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, inputs):
        return self.block(inputs)


class LegacyNavisVision(nn.Module):
    """Original 7x7 NavisVision architecture kept for checkpoint compatibility."""

    def __init__(self, num_classes=len(CLASS_NAMES), grid_size=7):
        super().__init__()
        self.num_classes = int(num_classes)
        self.grid_size = int(grid_size)
        self.backbone = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 192),
            ConvBlock(192, 256),
        )
        self.grid_pool = nn.AdaptiveAvgPool2d(
            (self.grid_size, self.grid_size)
        )
        self.detection_head = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.15),
            nn.Conv2d(256, 5 + self.num_classes, 1),
        )

    def forward(self, images):
        features = self.backbone(images)
        features = self.grid_pool(features)
        predictions = self.detection_head(features)
        return predictions.permute(0, 2, 3, 1).contiguous()


class ResidualBlock(nn.Module):
    """ResNet-18 basic block used by the redesigned detector."""

    expansion = 1

    def __init__(
        self,
        input_channels,
        output_channels,
        stride=1,
        downsample=None,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            input_channels,
            output_channels,
            3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            output_channels,
            output_channels,
            3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.downsample = downsample

    def forward(self, inputs):
        identity = inputs
        outputs = self.relu(self.bn1(self.conv1(inputs)))
        outputs = self.bn2(self.conv2(outputs))
        if self.downsample is not None:
            identity = self.downsample(inputs)
        return self.relu(outputs + identity)


class NavisVision(nn.Module):
    """Redesigned 13x13 NavisVision detector used by the Colab v2 notebook."""

    architecture_version = 2

    def __init__(self, num_classes=len(CLASS_NAMES), grid_size=13):
        super().__init__()
        self.num_classes = int(num_classes)
        self.grid_size = int(grid_size)
        self._backbone_channels = 64

        self.conv1 = nn.Conv2d(
            3,
            64,
            7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, blocks=2, stride=1)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, blocks=2, stride=2)

        self.grid_pool = nn.AdaptiveAvgPool2d(
            (self.grid_size, self.grid_size)
        )
        self.detection_head = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.10),
            nn.Conv2d(256, 5 + self.num_classes, 1),
        )

    def _make_layer(self, output_channels, blocks, stride):
        downsample = None
        if stride != 1 or self._backbone_channels != output_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self._backbone_channels,
                    output_channels,
                    1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(output_channels),
            )

        layers = [
            ResidualBlock(
                self._backbone_channels,
                output_channels,
                stride=stride,
                downsample=downsample,
            )
        ]
        self._backbone_channels = output_channels
        layers.extend(
            ResidualBlock(output_channels, output_channels)
            for _ in range(1, blocks)
        )
        return nn.Sequential(*layers)

    def forward(self, images):
        features = self.relu(self.bn1(self.conv1(images)))
        features = self.maxpool(features)
        features = self.layer1(features)
        features = self.layer2(features)
        features = self.layer3(features)
        features = self.grid_pool(features)
        predictions = self.detection_head(features)
        return predictions.permute(0, 2, 3, 1).contiguous()


class ConvBNAct(nn.Sequential):
    """Convolution, batch normalization, and SiLU used by NavisVision v3."""

    def __init__(
        self,
        input_channels,
        output_channels,
        kernel_size=3,
        stride=1,
    ):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.SiLU(inplace=True),
        )


class V3ResidualBlock(nn.Module):
    """Compact bottleneck residual block used by the v3 backbone."""

    def __init__(self, channels):
        super().__init__()
        hidden_channels = max(channels // 2, 16)
        self.block = nn.Sequential(
            ConvBNAct(channels, hidden_channels, kernel_size=1),
            ConvBNAct(hidden_channels, channels, kernel_size=3),
        )

    def forward(self, features):
        return features + self.block(features)


class DecoupledDetectionHead(nn.Module):
    """Separate box/objectness and coarse-class prediction towers."""

    def __init__(self, input_channels, num_classes):
        super().__init__()
        self.box_tower = nn.Sequential(
            ConvBNAct(input_channels, 128),
            ConvBNAct(128, 128),
        )
        self.class_tower = nn.Sequential(
            ConvBNAct(input_channels, 128),
            ConvBNAct(128, 128),
        )
        self.box_output = nn.Conv2d(128, 4, 1)
        self.object_output = nn.Conv2d(128, 1, 1)
        self.class_output = nn.Conv2d(128, num_classes, 1)

    def forward(self, features):
        box_features = self.box_tower(features)
        class_features = self.class_tower(features)
        predictions = torch.cat(
            (
                self.box_output(box_features),
                self.object_output(box_features),
                self.class_output(class_features),
            ),
            dim=1,
        )
        return predictions.permute(0, 2, 3, 1).contiguous()


class CropClassifier(nn.Module):
    """Small CNN that classifies an enlarged detected object crop."""

    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            ConvBNAct(3, 32, stride=2),
            ConvBNAct(32, 64, stride=2),
            V3ResidualBlock(64),
            ConvBNAct(64, 128, stride=2),
            V3ResidualBlock(128),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.Dropout(0.20),
            nn.Linear(128, num_classes),
        )

    def forward(self, crops):
        return self.classifier(self.features(crops))


class NavisVisionV3(nn.Module):
    """From-scratch multi-scale FPN/PAN detector with crop classifiers."""

    architecture_version = 3

    def __init__(
        self,
        detection_class_names=("traffic_light", "speed_sign"),
        fine_class_names=TRAFFIC_CONTROL_CLASS_NAMES,
        grid_sizes=V3_DEFAULT_GRID_SIZES,
        crop_size=96,
    ):
        super().__init__()
        self.detection_class_names = tuple(detection_class_names)
        self.fine_class_names = tuple(fine_class_names)
        self.num_detection_classes = len(self.detection_class_names)
        self.num_classes = len(self.fine_class_names)
        self.grid_sizes = tuple(int(value) for value in grid_sizes)
        self.crop_size = int(crop_size)
        self.group_fine_ids = {
            group_name: tuple(
                fine_id
                for fine_id, fine_name in enumerate(self.fine_class_names)
                if FINE_TO_DETECTION_GROUP[fine_name] == group_name
            )
            for group_name in self.detection_class_names
        }

        self.stem = ConvBNAct(3, 32, stride=2)
        self.stage2 = nn.Sequential(
            ConvBNAct(32, 64, stride=2),
            V3ResidualBlock(64),
        )
        self.stage3 = nn.Sequential(
            ConvBNAct(64, 128, stride=2),
            V3ResidualBlock(128),
            V3ResidualBlock(128),
        )
        self.stage4 = nn.Sequential(
            ConvBNAct(128, 256, stride=2),
            V3ResidualBlock(256),
            V3ResidualBlock(256),
        )
        self.stage5 = nn.Sequential(
            ConvBNAct(256, 512, stride=2),
            V3ResidualBlock(512),
            V3ResidualBlock(512),
        )

        self.p5_lateral = ConvBNAct(512, 256, kernel_size=1)
        self.p4_lateral = ConvBNAct(256, 256, kernel_size=1)
        self.p3_lateral = ConvBNAct(128, 256, kernel_size=1)
        self.p4_fuse = ConvBNAct(256, 256)
        self.p3_fuse = ConvBNAct(256, 256)
        self.pan4_down = ConvBNAct(256, 256, stride=2)
        self.pan4_fuse = ConvBNAct(256, 256)
        self.pan5_down = ConvBNAct(256, 256, stride=2)
        self.pan5_fuse = ConvBNAct(256, 256)

        self.detection_heads = nn.ModuleList(
            DecoupledDetectionHead(256, self.num_detection_classes)
            for _ in self.grid_sizes
        )
        self.crop_classifiers = nn.ModuleDict(
            {
                group_name: CropClassifier(len(fine_ids))
                for group_name, fine_ids in self.group_fine_ids.items()
                if len(fine_ids) > 1
            }
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(
                    module.weight,
                    nonlinearity="linear",
                )
                nn.init.zeros_(module.bias)

        object_prior_bias = float(
            torch.log(torch.tensor(0.01 / 0.99))
        )
        for detection_head in self.detection_heads:
            nn.init.constant_(
                detection_head.object_output.bias,
                object_prior_bias,
            )

    def forward(self, images):
        features = self.stem(images)
        features = self.stage2(features)
        c3 = self.stage3(features)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)

        p5 = self.p5_lateral(c5)
        p4 = self.p4_fuse(
            self.p4_lateral(c4)
            + F.interpolate(
                p5,
                size=c4.shape[-2:],
                mode="nearest",
            )
        )
        p3 = self.p3_fuse(
            self.p3_lateral(c3)
            + F.interpolate(
                p4,
                size=c3.shape[-2:],
                mode="nearest",
            )
        )
        n4 = self.pan4_fuse(p4 + self.pan4_down(p3))
        n5 = self.pan5_fuse(p5 + self.pan5_down(n4))
        pyramid = (p3, n4, n5)
        return [
            detection_head(scale_features)
            for detection_head, scale_features in zip(
                self.detection_heads,
                pyramid,
            )
        ]


def load_navisvision_checkpoint(checkpoint_path, device=None):
    """Return the checkpoint's model, metadata, and inference device."""

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"NavisVision checkpoint not found: {checkpoint_path}"
        )

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
    architecture = str(checkpoint.get("architecture", "")).lower()
    is_fasterrcnn = architecture == "fasterrcnn_resnet50_fpn_v2"
    if (
        not is_fasterrcnn
        and checkpoint.get("model_name", "NavisVision") != "NavisVision"
    ):
        raise ValueError(
            "Expected a NavisVision checkpoint, got "
            f"{checkpoint.get('model_name')!r}."
        )

    class_names = tuple(checkpoint.get("classes", CLASS_NAMES))
    if (
        not class_names
        or len(set(class_names)) != len(class_names)
        or any(class_name not in CLASS_NAMES for class_name in class_names)
    ):
        raise ValueError(
            "Checkpoint classes must be a non-empty, unique subset of the "
            "known NavisVision classes."
        )

    if is_fasterrcnn:
        image_size = int(checkpoint.get("image_size", 416))
        if image_size <= 0:
            raise ValueError(
                f"Invalid checkpoint image size: {image_size}."
            )
        try:
            from torchvision.models.detection import (
                fasterrcnn_resnet50_fpn_v2,
            )
            from torchvision.models.detection.faster_rcnn import (
                FastRCNNPredictor,
            )
        except ImportError as error:
            raise RuntimeError(
                "Faster R-CNN checkpoints require torchvision. Install the "
                "project requirements with 'py -3.12 -m pip install -r "
                "requirements.txt'."
            ) from error

        # The checkpoint contains the complete pretrained weights. Constructing
        # with weights=None prevents an unnecessary internet download in CARLA.
        model = fasterrcnn_resnet50_fpn_v2(
            weights=None,
            weights_backbone=None,
            min_size=image_size,
            max_size=image_size,
        )
        input_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(
            input_features,
            len(class_names) + 1,
        )
        model = model.to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        return model, checkpoint, device

    architecture_version = int(checkpoint.get("architecture_version", 1))
    if architecture_version == 1:
        model_type = LegacyNavisVision
        default_grid_size = 7
        default_image_size = 224
    elif architecture_version == 2:
        model_type = NavisVision
        default_grid_size = 13
        default_image_size = 416
    elif architecture_version == 3:
        model_type = NavisVisionV3
        default_grid_size = None
        default_image_size = 416
    else:
        raise ValueError(
            "Unsupported NavisVision architecture version: "
            f"{architecture_version}."
        )

    image_size = int(checkpoint.get("image_size", default_image_size))
    if image_size <= 0:
        raise ValueError(
            f"Invalid checkpoint image size: {image_size}."
        )

    if architecture_version == 3:
        detection_class_names = tuple(
            checkpoint.get(
                "detection_classes",
                tuple(
                    dict.fromkeys(
                        FINE_TO_DETECTION_GROUP[name]
                        for name in class_names
                    )
                ),
            )
        )
        expected_detection_classes = tuple(
            dict.fromkeys(
                FINE_TO_DETECTION_GROUP[name]
                for name in class_names
            )
        )
        if detection_class_names != expected_detection_classes:
            raise ValueError(
                "Checkpoint coarse detection classes do not match its fine "
                f"classes: {detection_class_names} versus "
                f"{expected_detection_classes}."
            )

        grid_sizes = tuple(
            int(value)
            for value in checkpoint.get(
                "grid_sizes",
                V3_DEFAULT_GRID_SIZES,
            )
        )
        crop_size = int(checkpoint.get("crop_size", 96))
        if (
            len(grid_sizes) != 3
            or any(grid_size <= 0 for grid_size in grid_sizes)
            or crop_size <= 0
        ):
            raise ValueError(
                "Invalid NavisVision v3 grid_sizes/crop_size metadata: "
                f"{grid_sizes}, {crop_size}."
            )
        model = model_type(
            detection_class_names=detection_class_names,
            fine_class_names=class_names,
            grid_sizes=grid_sizes,
            crop_size=crop_size,
        ).to(device)
    else:
        grid_size = int(
            checkpoint.get("grid_size", default_grid_size)
        )
        if grid_size <= 0:
            raise ValueError(
                f"Invalid checkpoint grid size: {grid_size}."
            )
        model = model_type(
            num_classes=len(class_names),
            grid_size=grid_size,
        ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint, device


def _box_iou(one_box, boxes):
    """Calculate IoU between one xyxy box and an array of xyxy boxes."""

    top_left = torch.maximum(one_box[:2], boxes[:, :2])
    bottom_right = torch.minimum(one_box[2:], boxes[:, 2:])
    intersection_size = (bottom_right - top_left).clamp_min(0.0)
    intersection = intersection_size[:, 0] * intersection_size[:, 1]

    one_size = (one_box[2:] - one_box[:2]).clamp_min(0.0)
    one_area = one_size[0] * one_size[1]
    box_sizes = (boxes[:, 2:] - boxes[:, :2]).clamp_min(0.0)
    box_areas = box_sizes[:, 0] * box_sizes[:, 1]
    union = one_area + box_areas - intersection
    return intersection / union.clamp_min(1e-12)


def _nms(boxes, scores, iou_threshold):
    """Pure-PyTorch non-maximum suppression, avoiding a torchvision dependency."""

    if len(boxes) == 0:
        return torch.empty((0,), dtype=torch.long)

    order = scores.argsort(descending=True)
    kept = []
    while len(order):
        current = order[0]
        kept.append(int(current))
        if len(order) == 1:
            break
        remaining = order[1:]
        ious = _box_iou(boxes[current], boxes[remaining])
        order = remaining[ious <= iou_threshold]
    return torch.tensor(kept, dtype=torch.long)


class TemporalDetectionFilter:
    """Confirm, track, and smooth detections across consecutive video frames."""

    def __init__(
        self,
        class_names,
        window_size=5,
        minimum_hits=3,
        match_iou=0.30,
        max_missed=2,
        smoothing=0.35,
    ):
        self.class_names = tuple(class_names)
        self.window_size = int(window_size)
        self.minimum_hits = int(minimum_hits)
        self.match_iou = float(match_iou)
        self.max_missed = int(max_missed)
        self.smoothing = float(smoothing)
        if self.window_size < 1:
            raise ValueError("Temporal window_size must be at least 1.")
        if not 1 <= self.minimum_hits <= self.window_size:
            raise ValueError(
                "Temporal minimum_hits must be between 1 and window_size."
            )
        if not 0.0 <= self.match_iou <= 1.0:
            raise ValueError("Temporal match_iou must be between 0 and 1.")
        if self.max_missed < 0:
            raise ValueError("Temporal max_missed cannot be negative.")
        if not 0.0 < self.smoothing <= 1.0:
            raise ValueError("Temporal smoothing must be above 0 and at most 1.")
        self.reset()

    def reset(self):
        """Discard tracks, such as after the CARLA vehicle or camera respawns."""

        self.tracks = []

    @staticmethod
    def _class_group(label):
        # Different light colours and speed-limit values describe the same
        # physical object, so allow their class votes to share one track.
        if label.startswith("traffic_light_"):
            return "traffic_light"
        if label.startswith("traffic_sign_"):
            return "traffic_sign"
        return label

    @staticmethod
    def _iou(one_box, another_box):
        left = max(one_box[0], another_box[0])
        top = max(one_box[1], another_box[1])
        right = min(one_box[2], another_box[2])
        bottom = min(one_box[3], another_box[3])
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        one_area = max(0.0, one_box[2] - one_box[0]) * max(
            0.0, one_box[3] - one_box[1]
        )
        another_area = max(
            0.0, another_box[2] - another_box[0]
        ) * max(0.0, another_box[3] - another_box[1])
        union = one_area + another_area - intersection
        return intersection / max(union, 1e-12)

    def _winning_class(self, track):
        votes = {}
        for class_id, score in track["class_votes"]:
            votes[class_id] = votes.get(class_id, 0.0) + score
        return max(votes, key=votes.get)

    def update(self, detections):
        """Return only spatially stable detections confirmed over time."""

        for track in self.tracks:
            track["hits"].append(False)
            track["missed"] += 1

        unmatched_track_ids = set(range(len(self.tracks)))
        for detection in sorted(
            detections,
            key=lambda item: item["score"],
            reverse=True,
        ):
            detection_box = np.asarray(detection["box"], dtype=np.float32)
            detection_group = self._class_group(detection["label"])
            best_track_id = None
            best_iou = self.match_iou
            for track_id in unmatched_track_ids:
                track = self.tracks[track_id]
                if track["group"] != detection_group:
                    continue
                overlap = self._iou(track["box"], detection_box)
                if overlap >= best_iou:
                    best_iou = overlap
                    best_track_id = track_id

            if best_track_id is None:
                self.tracks.append(
                    {
                        "box": detection_box,
                        "score": float(detection["score"]),
                        "group": detection_group,
                        "hits": deque(
                            [True],
                            maxlen=self.window_size,
                        ),
                        "class_votes": deque(
                            [
                                (
                                    int(detection["class_id"]),
                                    float(detection["score"]),
                                )
                            ],
                            maxlen=self.window_size,
                        ),
                        "missed": 0,
                    }
                )
                continue

            track = self.tracks[best_track_id]
            unmatched_track_ids.remove(best_track_id)
            track["box"] = (
                (1.0 - self.smoothing) * track["box"]
                + self.smoothing * detection_box
            )
            track["score"] = (
                (1.0 - self.smoothing) * track["score"]
                + self.smoothing * float(detection["score"])
            )
            track["hits"][-1] = True
            track["class_votes"].append(
                (
                    int(detection["class_id"]),
                    float(detection["score"]),
                )
            )
            track["missed"] = 0

        self.tracks = [
            track
            for track in self.tracks
            if track["missed"] <= self.max_missed
        ]

        confirmed = []
        for track in self.tracks:
            if sum(track["hits"]) < self.minimum_hits:
                continue
            class_id = self._winning_class(track)
            confirmed.append(
                {
                    "box": tuple(float(value) for value in track["box"]),
                    "score": float(track["score"]),
                    "class_id": class_id,
                    "label": self.class_names[class_id],
                }
            )
        return sorted(
            confirmed,
            key=lambda detection: detection["score"],
            reverse=True,
        )


class NavisVisionRuntime:
    """Preprocess CARLA RGB frames and return decoded object detections."""

    def __init__(
        self,
        checkpoint_path,
        confidence_threshold=0.50,
        nms_threshold=0.40,
        device=None,
        temporal_filter=True,
        temporal_window=5,
        temporal_minimum_hits=3,
        temporal_match_iou=0.30,
        temporal_max_missed=2,
        temporal_smoothing=0.35,
    ):
        self.model, self.metadata, self.device = load_navisvision_checkpoint(
            checkpoint_path,
            device=device,
        )
        self.detector_family = str(
            self.metadata.get("architecture", "navisvision")
        ).lower()
        self.is_fasterrcnn = (
            self.detector_family == "fasterrcnn_resnet50_fpn_v2"
        )
        self.architecture_version = (
            None
            if self.is_fasterrcnn
            else int(self.metadata.get("architecture_version", 1))
        )
        self.class_names = tuple(self.metadata.get("classes", CLASS_NAMES))
        if self.is_fasterrcnn:
            self.grid_size = None
            self.grid_sizes = ()
            self.detection_class_names = self.class_names
        elif self.architecture_version == 3:
            self.grid_sizes = tuple(
                int(value)
                for value in self.metadata.get(
                    "grid_sizes",
                    self.model.grid_sizes,
                )
            )
            self.detection_class_names = tuple(
                self.metadata.get(
                    "detection_classes",
                    self.model.detection_class_names,
                )
            )
            self.grid_size = None
        else:
            self.grid_size = int(
                self.metadata.get("grid_size", self.model.grid_size)
            )
            self.grid_sizes = (self.grid_size,)
            self.detection_class_names = self.class_names
        self.image_size = int(self.metadata.get("image_size", 224))
        self.confidence_threshold = float(confidence_threshold)
        self.nms_threshold = float(nms_threshold)
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1.")
        if not 0.0 <= self.nms_threshold <= 1.0:
            raise ValueError("nms_threshold must be between 0 and 1.")
        if self.is_fasterrcnn:
            # Torchvision performs score filtering and NMS inside the detector.
            # Match those settings to the existing command-line arguments.
            self.model.roi_heads.score_thresh = self.confidence_threshold
            self.model.roi_heads.nms_thresh = self.nms_threshold
        self.temporal_filter = (
            TemporalDetectionFilter(
                self.class_names,
                window_size=temporal_window,
                minimum_hits=temporal_minimum_hits,
                match_iou=temporal_match_iou,
                max_missed=temporal_max_missed,
                smoothing=temporal_smoothing,
            )
            if temporal_filter
            else None
        )

    def _preprocess(self, rgb_array):
        rgb_array = np.asarray(rgb_array)
        if rgb_array.ndim != 3 or rgb_array.shape[2] != 3:
            raise ValueError(
                "NavisVision expects an RGB frame shaped [height, width, 3]."
            )
        tensor = torch.from_numpy(
            np.ascontiguousarray(rgb_array)
        ).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.device, non_blocking=True).float().div_(255.0)
        return F.interpolate(
            tensor,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

    @staticmethod
    def _cxcywh_to_xyxy(center_x, center_y, width, height):
        return torch.stack(
            (
                center_x - width / 2,
                center_y - height / 2,
                center_x + width / 2,
                center_y + height / 2,
            ),
            dim=-1,
        ).clamp(0.0, 1.0)

    def _decode_single_scale(self, raw_predictions):
        predictions = raw_predictions.detach().float().cpu()[0]
        grid_size = predictions.shape[0]
        grid_y, grid_x = torch.meshgrid(
            torch.arange(grid_size),
            torch.arange(grid_size),
            indexing="ij",
        )
        center_x = (predictions[..., 0].sigmoid() + grid_x) / grid_size
        center_y = (predictions[..., 1].sigmoid() + grid_y) / grid_size
        width = predictions[..., 2].sigmoid()
        height = predictions[..., 3].sigmoid()
        boxes = self._cxcywh_to_xyxy(
            center_x,
            center_y,
            width,
            height,
        ).reshape(-1, 4)

        objectness = predictions[..., 4].sigmoid()
        class_probabilities = predictions[..., 5:].softmax(dim=-1)
        class_confidence, class_ids = class_probabilities.max(dim=-1)
        scores = (objectness * class_confidence).reshape(-1)
        class_ids = class_ids.reshape(-1)

        score_mask = scores >= self.confidence_threshold
        boxes = boxes[score_mask]
        scores = scores[score_mask]
        class_ids = class_ids[score_mask]

        # Suppress overlaps across every class. This prevents one physical
        # object from simultaneously receiving several incompatible labels.
        kept_positions = _nms(
            boxes,
            scores,
            self.nms_threshold,
        ).tolist()

        kept_positions.sort(key=lambda index: float(scores[index]), reverse=True)
        detections = []
        for index in kept_positions:
            class_id = int(class_ids[index])
            detections.append(
                {
                    "box": tuple(float(value) for value in boxes[index]),
                    "score": float(scores[index]),
                    "class_id": class_id,
                    "label": self.class_names[class_id],
                }
            )
        return detections

    def _decode_fasterrcnn(self, output):
        """Convert torchvision absolute xyxy output to normalized detections."""

        boxes = output["boxes"].detach().float().cpu()
        scores = output["scores"].detach().float().cpu()
        labels = output["labels"].detach().cpu()
        score_mask = scores >= self.confidence_threshold
        boxes = boxes[score_mask]
        scores = scores[score_mask]
        labels = labels[score_mask]

        detections = []
        for box, score, label in zip(boxes, scores, labels):
            # Torchvision reserves label 0 for background. The checkpoint's
            # six traffic-control classes therefore use labels 1 through 6.
            class_id = int(label) - 1
            if not 0 <= class_id < len(self.class_names):
                continue
            normalized_box = (box / float(self.image_size)).clamp(0.0, 1.0)
            detections.append(
                {
                    "box": tuple(float(value) for value in normalized_box),
                    "score": float(score),
                    "class_id": class_id,
                    "label": self.class_names[class_id],
                }
            )
        return detections

    def _decode_v3_coarse(self, raw_predictions_by_scale):
        """Merge v3 P3/P4/P5 outputs into coarse normalized detections."""

        all_boxes = []
        all_scores = []
        all_class_ids = []

        for raw_predictions in raw_predictions_by_scale:
            predictions = raw_predictions.detach().float().cpu()[0]
            grid_size = predictions.shape[0]
            grid_y, grid_x = torch.meshgrid(
                torch.arange(grid_size),
                torch.arange(grid_size),
                indexing="ij",
            )
            center_x = (
                predictions[..., 0].sigmoid() + grid_x
            ) / grid_size
            center_y = (
                predictions[..., 1].sigmoid() + grid_y
            ) / grid_size
            width = predictions[..., 2].sigmoid()
            height = predictions[..., 3].sigmoid()
            boxes = self._cxcywh_to_xyxy(
                center_x,
                center_y,
                width,
                height,
            ).reshape(-1, 4)

            objectness = predictions[..., 4].sigmoid()
            class_probabilities = predictions[..., 5:].softmax(dim=-1)
            class_confidence, class_ids = class_probabilities.max(dim=-1)
            scores = (objectness * class_confidence).reshape(-1)
            class_ids = class_ids.reshape(-1)
            score_mask = scores >= self.confidence_threshold
            all_boxes.append(boxes[score_mask])
            all_scores.append(scores[score_mask])
            all_class_ids.append(class_ids[score_mask])

        boxes = torch.cat(all_boxes, dim=0)
        scores = torch.cat(all_scores, dim=0)
        class_ids = torch.cat(all_class_ids, dim=0)
        if len(boxes) == 0:
            return []

        # Bound quadratic pure-PyTorch NMS and crop-classifier work if a weak
        # checkpoint produces an unusually large number of candidates.
        if len(scores) > 300:
            top_positions = scores.topk(300).indices
            boxes = boxes[top_positions]
            scores = scores[top_positions]
            class_ids = class_ids[top_positions]

        kept_positions = _nms(
            boxes,
            scores,
            self.nms_threshold,
        ).tolist()[:100]
        kept_positions.sort(
            key=lambda index: float(scores[index]),
            reverse=True,
        )
        return [
            {
                "box": tuple(float(value) for value in boxes[index]),
                "score": float(scores[index]),
                "coarse_class_id": int(class_ids[index]),
            }
            for index in kept_positions
        ]

    def _extract_v3_crops(self, inputs, boxes):
        """Bilinearly sample normalized boxes without requiring torchvision."""

        crop_size = int(self.model.crop_size)
        unit_x = torch.linspace(
            0.0,
            1.0,
            crop_size,
            device=self.device,
        )
        unit_y = torch.linspace(
            0.0,
            1.0,
            crop_size,
            device=self.device,
        )
        grids = []
        for box in boxes:
            left, top, right, bottom = (
                torch.tensor(
                    box,
                    device=self.device,
                    dtype=inputs.dtype,
                ).clamp(0.0, 1.0)
            )
            x_coordinates = left + unit_x * (right - left)
            y_coordinates = top + unit_y * (bottom - top)
            grid_y, grid_x = torch.meshgrid(
                y_coordinates,
                x_coordinates,
                indexing="ij",
            )
            grids.append(
                torch.stack(
                    (
                        grid_x * 2.0 - 1.0,
                        grid_y * 2.0 - 1.0,
                    ),
                    dim=-1,
                )
            )

        sampling_grid = torch.stack(grids)
        source_images = inputs.expand(len(grids), -1, -1, -1)
        return F.grid_sample(
            source_images,
            sampling_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

    def _classify_v3_detections(self, inputs, coarse_detections):
        """Recover final light colour or speed value from detected crops."""

        if not coarse_detections:
            return []

        grouped_positions = {}
        for detection_index, detection in enumerate(coarse_detections):
            coarse_id = detection["coarse_class_id"]
            group_name = self.detection_class_names[coarse_id]
            grouped_positions.setdefault(group_name, []).append(
                detection_index
            )

        detections = []
        for group_name, positions in grouped_positions.items():
            fine_ids = self.model.group_fine_ids[group_name]
            if group_name in self.model.crop_classifiers:
                boxes = [
                    coarse_detections[position]["box"]
                    for position in positions
                ]
                crops = self._extract_v3_crops(inputs, boxes)
                probabilities = self.model.crop_classifiers[
                    group_name
                ](crops).softmax(dim=1)
                fine_confidences, local_class_ids = probabilities.max(
                    dim=1
                )
                fine_confidences = fine_confidences.detach().float().cpu()
                local_class_ids = local_class_ids.detach().cpu()
            else:
                fine_confidences = torch.ones(len(positions))
                local_class_ids = torch.zeros(
                    len(positions),
                    dtype=torch.long,
                )

            for output_index, detection_index in enumerate(positions):
                coarse_detection = coarse_detections[detection_index]
                fine_class_id = fine_ids[int(local_class_ids[output_index])]
                final_score = (
                    coarse_detection["score"]
                    * float(fine_confidences[output_index])
                )
                if final_score < self.confidence_threshold:
                    continue
                detections.append(
                    {
                        "box": coarse_detection["box"],
                        "score": final_score,
                        "class_id": fine_class_id,
                        "label": self.class_names[fine_class_id],
                    }
                )

        return sorted(
            detections,
            key=lambda detection: detection["score"],
            reverse=True,
        )

    def predict(self, rgb_array):
        """Return normalized xyxy detections for one CARLA RGB frame."""

        inputs = self._preprocess(rgb_array)
        with torch.inference_mode():
            if self.is_fasterrcnn:
                raw_predictions = self.model([inputs[0]])
                detections = self._decode_fasterrcnn(raw_predictions[0])
            else:
                raw_predictions = self.model(inputs)
            if not self.is_fasterrcnn and self.architecture_version == 3:
                coarse_detections = self._decode_v3_coarse(
                    raw_predictions
                )
                detections = self._classify_v3_detections(
                    inputs,
                    coarse_detections,
                )
            elif not self.is_fasterrcnn:
                detections = self._decode_single_scale(raw_predictions)
        if self.temporal_filter is not None:
            detections = self.temporal_filter.update(detections)
        return detections

    def reset_temporal_filter(self):
        """Clear temporal state after a camera or vehicle change."""

        if self.temporal_filter is not None:
            self.temporal_filter.reset()


def load_traffic_control_classifier_checkpoint(checkpoint_path, device=None):
    """Load a transfer-learned ResNet-18 or ViT-B/16 crop classifier."""

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Traffic-control classifier checkpoint not found: {checkpoint_path}"
        )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(
            "Classifier checkpoint must contain a 'state_dict'. Use the "
            "checkpoint saved by the ResNet-18 or ViT-B/16 Colab cell."
        )

    task = checkpoint.get("task")
    if task != "traffic_control_crop_classification":
        raise ValueError(
            "Expected a traffic-control crop-classification checkpoint, got "
            f"task={task!r}."
        )

    class_names = tuple(checkpoint.get("classes", TRAFFIC_CONTROL_CLASS_NAMES))
    if not class_names:
        raise ValueError("Classifier checkpoint contains no class names.")

    architecture = str(checkpoint.get("architecture", "")).lower()
    try:
        if "resnet18" in architecture:
            from torchvision.models import resnet18

            model = resnet18(weights=None)
            input_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(p=0.25),
                nn.Linear(input_features, len(class_names)),
            )
        elif "vit_b_16" in architecture or "vit-b/16" in architecture:
            from torchvision.models import vit_b_16

            model = vit_b_16(weights=None)
            input_features = model.heads.head.in_features
            model.heads = nn.Sequential(
                nn.Dropout(p=0.25),
                nn.Linear(input_features, len(class_names)),
            )
        else:
            raise ValueError(
                "Unsupported classifier architecture: "
                f"{checkpoint.get('architecture')!r}."
            )
    except ImportError as error:
        raise RuntimeError(
            "The classifier runtime requires torchvision. Install it in the "
            "same Python 3.12 environment used to launch CARLA."
        ) from error

    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    return model, checkpoint, device


class TrafficControlClassifierRuntime:
    """Independently classify the crops produced by NavisVision boxes."""

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, checkpoint_path, device=None, context=0.20):
        self.model, self.metadata, self.device = (
            load_traffic_control_classifier_checkpoint(
                checkpoint_path,
                device=device,
            )
        )
        self.class_names = tuple(
            self.metadata.get("classes", TRAFFIC_CONTROL_CLASS_NAMES)
        )
        self.class_name_set = set(self.class_names)
        self.image_size = int(self.metadata.get("image_size", 224))
        self.context = float(self.metadata.get("crop_context", context))
        if self.image_size <= 0:
            raise ValueError("Classifier image_size must be positive.")
        if self.context < 0.0:
            raise ValueError("Classifier crop context cannot be negative.")

        self.mean = torch.tensor(
            self.IMAGENET_MEAN,
            device=self.device,
        ).view(1, 3, 1, 1)
        self.std = torch.tensor(
            self.IMAGENET_STD,
            device=self.device,
        ).view(1, 3, 1, 1)

    def _crop_tensor(self, rgb_array, box):
        image_height, image_width = rgb_array.shape[:2]
        left, top, right, bottom = (
            float(value) for value in box
        )
        left *= image_width
        right *= image_width
        top *= image_height
        bottom *= image_height

        pad_x = max(2.0, (right - left) * self.context)
        pad_y = max(2.0, (bottom - top) * self.context)
        left = max(0, int(round(left - pad_x)))
        right = min(image_width, int(round(right + pad_x)))
        top = max(0, int(round(top - pad_y)))
        bottom = min(image_height, int(round(bottom + pad_y)))
        if right <= left or bottom <= top:
            return None

        crop = np.ascontiguousarray(rgb_array[top:bottom, left:right])
        tensor = torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.device, non_blocking=True).float().div_(255.0)
        return F.interpolate(
            tensor,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

    def classify_detections(self, rgb_array, detections):
        """Attach classifier labels and scores to eligible detections."""

        rgb_array = np.asarray(rgb_array)
        if rgb_array.ndim != 3 or rgb_array.shape[2] != 3:
            raise ValueError(
                "The crop classifier expects an RGB frame shaped "
                "[height, width, 3]."
            )

        enriched = [dict(detection) for detection in detections]
        crop_tensors = []
        detection_positions = []
        for position, detection in enumerate(enriched):
            if detection.get("label") not in self.class_name_set:
                continue
            crop_tensor = self._crop_tensor(rgb_array, detection["box"])
            if crop_tensor is None:
                continue
            crop_tensors.append(crop_tensor)
            detection_positions.append(position)

        if not crop_tensors:
            return enriched

        inputs = torch.cat(crop_tensors, dim=0)
        inputs = (inputs - self.mean) / self.std
        with torch.inference_mode():
            probabilities = self.model(inputs).softmax(dim=1)
        scores, class_ids = probabilities.max(dim=1)
        scores = scores.detach().float().cpu()
        class_ids = class_ids.detach().cpu()

        for output_position, detection_position in enumerate(
                detection_positions):
            class_id = int(class_ids[output_position])
            enriched[detection_position].update(
                {
                    "classifier_class_id": class_id,
                    "classifier_label": self.class_names[class_id],
                    "classifier_score": float(scores[output_position]),
                    "classifier_agrees": (
                        enriched[detection_position].get("label")
                        == self.class_names[class_id]
                    ),
                }
            )
        return enriched
