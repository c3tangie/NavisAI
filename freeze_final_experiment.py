"""Create a reproducible, read-only-style snapshot of the final NavisAI experiment.

This does not freeze neural-network layers. It copies the selected checkpoints and
source files into one dated directory and records SHA-256 hashes, checkpoint
metadata, runtime settings, and Git state in a manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent

# These are the final models currently selected for the report. The custom V2
# checkpoint must be downloaded from Google Drive using this exact filename.
FINAL_CHECKPOINTS = {
    "navissteer": "models/navissteer_finetuned_bestV2.pt",
    "linearsteer": "models/linearsteer_best.pt",
    "custom_navisvision_v2": "models/navisvision_v2_traffic_controls_best.pt",
    "faster_rcnn": "models/navisvision_fasterrcnn_resnet50_fpn_best.pt",
}

SOURCE_FILES = (
    "manual_control.py",
    "navissteer_model.py",
    "navisvision_model.py",
    "requirements.txt",
)

FINAL_SETTINGS = {
    "steering": {
        "source_frame_size": [220, 220],
        "roi_crop_xyxy": [0, 110, 220, 220],
        "model_input_chw": [3, 110, 220],
    },
    "object_detection": {
        "model_input_hw": [416, 416],
        "classes": [
            "traffic_light_green",
            "traffic_light_orange",
            "traffic_light_red",
            "traffic_sign_30",
            "traffic_sign_60",
            "traffic_sign_90",
        ],
        "offline_counting_confidence": 0.50,
        "carla_display_confidence": 0.40,
        "nms_iou": 0.40,
        "carla_detector_fps": 5.0,
        "temporal_filter": {
            "enabled": True,
            "window": 5,
            "minimum_hits": 3,
        },
    },
    "new_data_policy": {
        "training_allowed": False,
        "validation_allowed": False,
        "threshold_tuning_allowed": False,
        "purpose": "final evaluation only",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_metadata(path: Path) -> dict:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        payload = torch.load(path, map_location="cpu", weights_only=False)

    if not isinstance(payload, dict):
        return {"payload_type": type(payload).__name__}

    allowed = (
        "model_name",
        "architecture",
        "architecture_version",
        "task",
        "pretrained",
        "epoch",
        "validation_loss",
        "validation_mse",
        "validation_map_50",
        "image_size",
        "input_mode",
        "input_channels",
        "frame_stride",
        "base_frame_stride",
        "new_data_frame_stride",
        "recovery_data_frame_stride",
        "batch_size",
        "learning_rate",
        "split_seed",
        "classes",
    )
    metadata = {}
    for key in allowed:
        if key in payload:
            value = payload[key]
            if isinstance(value, tuple):
                value = list(value)
            metadata[key] = value
    return metadata


def git_information() -> dict:
    def run_git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    return {
        "commit": run_git("rev-parse", "HEAD"),
        "working_tree_status_at_freeze": run_git("status", "--short").splitlines(),
    }


def copy_and_describe(source: Path, destination: Path) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return {
        "source": str(source.relative_to(PROJECT_ROOT)),
        "frozen_copy": str(destination.relative_to(destination.parents[1])),
        "bytes": destination.stat().st_size,
        "sha256": sha256(destination),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "final_experiment_2026-08-05",
        help="snapshot directory",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing snapshot directory",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="create a partial snapshot while reporting missing files",
    )
    args = parser.parse_args()
    output = args.output.resolve()

    missing = [
        relative
        for relative in (*FINAL_CHECKPOINTS.values(), *SOURCE_FILES)
        if not (PROJECT_ROOT / relative).is_file()
    ]
    if missing and not args.allow_incomplete:
        print("Cannot freeze the complete experiment. Missing files:")
        for relative in missing:
            print(f"  - {relative}")
        print("Download the missing checkpoint(s), then run this command again.")
        return 2

    if output.exists():
        if not args.force:
            print(f"Snapshot already exists: {output}")
            print("Use --force only if you intentionally want to replace it.")
            return 3
        shutil.rmtree(output)

    output.mkdir(parents=True)
    manifest = {
        "experiment_status": "complete" if not missing else "INCOMPLETE",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root_at_freeze": str(PROJECT_ROOT),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        },
        "git": git_information(),
        "settings": FINAL_SETTINGS,
        "missing_files": missing,
        "checkpoints": {},
        "source_files": {},
    }

    for model_name, relative in FINAL_CHECKPOINTS.items():
        source = PROJECT_ROOT / relative
        if not source.is_file():
            manifest["checkpoints"][model_name] = {
                "status": "MISSING",
                "expected_path": relative,
            }
            continue
        destination = output / relative
        description = copy_and_describe(source, destination)
        description["metadata"] = checkpoint_metadata(destination)
        manifest["checkpoints"][model_name] = description

    for relative in SOURCE_FILES:
        source = PROJECT_ROOT / relative
        if source.is_file():
            destination = output / "source" / relative
            manifest["source_files"][relative] = copy_and_describe(
                source, destination
            )

    manifest_path = output / "experiment_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output / "DO_NOT_TUNE_ON_FINAL_NEW_DATA.txt").write_text(
        "This snapshot defines the final experiment. Data collected after this "
        "freeze is evaluation-only and must not be used for training, validation, "
        "confidence-threshold selection, NMS selection, or preprocessing changes.\n",
        encoding="utf-8",
    )

    print(f"Created snapshot: {output}")
    print(f"Status: {manifest['experiment_status']}")
    print(f"Manifest: {manifest_path}")
    if missing:
        print("Missing files:")
        for relative in missing:
            print(f"  - {relative}")
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
