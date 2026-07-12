import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split


IMAGE_COLUMNS = ("image", "image_path", "filename", "file", "center", "front", "rgb")
STEERING_COLUMNS = ("steering", "steering_angle", "angle", "steer")


class SteeringDataset(Dataset):
    def __init__(self, manifest, image_column, steering_column, width, height):
        self.manifest = manifest.reset_index(drop=True)
        self.image_column = image_column
        self.steering_column = steering_column
        self.width = width
        self.height = height

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, index):
        row = self.manifest.iloc[index]
        with Image.open(row[self.image_column]) as image:
            image = image.convert("L").resize((self.width, self.height))
            pixels = np.asarray(image, dtype=np.float32).reshape(-1) / 255.0
        steering = np.float32(row[self.steering_column])
        return torch.from_numpy(pixels), torch.tensor([steering])


class LinearSteeringModel(nn.Module):
    def __init__(self, width, height):
        super().__init__()
        self.linear = nn.Linear(width * height, 1)

    def forward(self, images):
        return self.linear(images)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a linear steering baseline.")
    parser.add_argument("--csv", type=Path, required=True, help="CSV containing image paths and steering values.")
    parser.add_argument("--image-root", type=Path, help="Base directory for relative image paths.")
    parser.add_argument("--image-column", help="CSV image-path column (auto-detected by default).")
    parser.add_argument("--steering-column", help="CSV steering column (auto-detected by default).")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=36)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("models/linear_steering.pt"))
    return parser.parse_args()


def find_column(columns, requested, candidates, kind):
    if requested:
        if requested not in columns:
            raise ValueError(f"{kind} column '{requested}' not found. Available: {list(columns)}")
        return requested
    normalized = {str(column).strip().lower(): column for column in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    raise ValueError(f"Could not detect the {kind} column. Pass --{kind}-column.")


def load_manifest(csv_path, image_root, image_column, steering_column):
    manifest = pd.read_csv(csv_path)
    image_column = find_column(manifest.columns, image_column, IMAGE_COLUMNS, "image")
    steering_column = find_column(manifest.columns, steering_column, STEERING_COLUMNS, "steering")
    manifest = manifest[[image_column, steering_column]].dropna().copy()
    manifest[steering_column] = pd.to_numeric(manifest[steering_column], errors="coerce")
    manifest = manifest.dropna()

    root = image_root or csv_path.parent

    def resolve_image(value):
        path = Path(str(value).strip())
        if path.is_absolute() and path.is_file():
            return path
        direct = root / path
        if direct.is_file():
            return direct
        filename_only = root / path.name
        return filename_only

    manifest[image_column] = manifest[image_column].map(resolve_image)
    missing = [path for path in manifest[image_column] if not path.is_file()]
    if missing:
        examples = ", ".join(str(path) for path in missing[:3])
        raise FileNotFoundError(f"Could not find {len(missing)} images. Examples: {examples}")
    if len(manifest) < 2:
        raise ValueError("At least two valid image/steering rows are required.")
    return manifest, image_column, steering_column


def evaluate(model, loader, device):
    model.eval()
    squared_error = 0.0
    absolute_error = 0.0
    count = 0
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            predictions = model(images)
            squared_error += torch.square(predictions - targets).sum().item()
            absolute_error += torch.abs(predictions - targets).sum().item()
            count += targets.numel()
    return squared_error / count, absolute_error / count


def main():
    args = parse_args()
    if not 0 < args.validation_fraction < 1:
        raise SystemExit("--validation-fraction must be between 0 and 1.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    manifest, image_column, steering_column = load_manifest(
        args.csv, args.image_root, args.image_column, args.steering_column
    )
    dataset = SteeringDataset(manifest, image_column, steering_column, args.width, args.height)
    validation_size = max(1, round(len(dataset) * args.validation_fraction))
    training_size = len(dataset) - validation_size
    if training_size == 0:
        raise SystemExit("The validation split leaves no training examples.")

    generator = torch.Generator().manual_seed(args.seed)
    training_data, validation_data = random_split(
        dataset, [training_size, validation_size], generator=generator
    )
    training_loader = DataLoader(
        training_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, generator=generator
    )
    validation_loader = DataLoader(
        validation_data, batch_size=args.batch_size, num_workers=args.num_workers
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LinearSteeringModel(args.width, args.height).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    loss_function = nn.MSELoss()

    print(f"Training on {training_size} images; validating on {validation_size}; device={device}")
    best_mse = float("inf")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        for images, targets in training_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = loss_function(model(images), targets)
            loss.backward()
            optimizer.step()

        validation_mse, validation_mae = evaluate(model, validation_loader, device)
        print(f"Epoch {epoch:03d}: val_mse={validation_mse:.6f} val_mae={validation_mae:.6f}")
        if validation_mse < best_mse:
            best_mse = validation_mse
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "width": args.width,
                    "height": args.height,
                    "image_mode": "L",
                    "pixel_scale": 255.0,
                    "validation_mse": validation_mse,
                    "validation_mae": validation_mae,
                },
                args.output,
            )

    metrics_path = args.output.with_suffix(".json")
    metrics_path.write_text(
        json.dumps({"best_validation_mse": best_mse, "samples": len(dataset)}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved best model to {args.output} and metrics to {metrics_path}")


if __name__ == "__main__":
    main()
