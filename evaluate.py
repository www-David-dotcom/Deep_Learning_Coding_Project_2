import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import Final

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.v2 import Compose, Normalize, ToDtype

from datasets import TinyImageNetDataset
from modules import CustomModel

DATASET_ROOT: Final = Path("data/tiny-imagenet-200")
DEVICE: Final = torch.accelerator.current_accelerator(
    check_available=True
) or torch.device("cpu")
TRANSFORM: Final = Compose(
    [
        ToDtype(torch.float32, scale=True),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    dataset: Dataset[tuple[Tensor, int]],
    *,
    batch_size: int = 1,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> float:
    model.eval()

    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory
    )

    num_correct, num_samples = 0, 0
    for images, labels in dataloader:
        images: Tensor
        labels: Tensor

        images, labels = images.to(DEVICE), labels.to(DEVICE)

        logits = model(images)
        preds = torch.argmax(logits, dim=1)

        num_correct += (preds == labels).sum().item()
        num_samples += labels.size(0)

    return num_correct / num_samples


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = ArgumentParser()
    parser.add_argument("ckpt_path", type=Path, help="Path to model checkpoint")
    args = parser.parse_args()

    ckpt_path: Path = args.ckpt_path

    model = CustomModel().to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))

    dataset = TinyImageNetDataset(DATASET_ROOT, "val", transform=TRANSFORM)

    accuracy = evaluate(
        model,
        dataset,
        batch_size=64,
        num_workers=4,
        pin_memory=torch.accelerator.is_available(),
    )

    logging.info(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
