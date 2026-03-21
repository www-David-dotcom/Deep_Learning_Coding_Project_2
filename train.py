import logging
import os
from argparse import ArgumentParser
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import (
    Compose,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    ToDtype,
)

from datasets import TinyImageNetDataset
from evaluate import DATASET_ROOT, DEVICE, TRANSFORM
from modules import CustomModel


class ModelEMA:
    """
    Exponential Moving average. It keeps a smoother copy of the weights
    """
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = decay
        # clone the state dict
        self.state_dict = { name: tensor.detach().clone() for name, tensor in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        model_state = model.state_dict()
        for name, ema_tensor in self.state_dict.items():
            model_tensor = model_state[name].detach()
            if torch.is_floating_point(ema_tensor):
                # lerp: linear interpolation x<-x + w(y-x)
                ema_tensor.lerp_(model_tensor, 1.0 - self.decay)
            else:
                ema_tensor.copy_(model_tensor)

    def copy_to(self, model: torch.nn.Module) -> None:
        model.load_state_dict(self.state_dict)


def cutmix_batch(images: Tensor, labels: Tensor, alpha: float) -> tuple[Tensor, Tensor, Tensor, float]:
    """
    augment the data by cutting one rectangle of an image to the other
    """

    # create a sample of a beta distribution, called \lambda
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    # do a random permutation
    index = torch.randperm(images.size(0), device=images.device)

    _, _, height, width = images.shape
    # cut a random rectangular area
    cut_ratio = (1.0 - lam) ** 0.5
    cut_w = int(width * cut_ratio)
    cut_h = int(height * cut_ratio)
    center_x = torch.randint(width, (1,), device=images.device).item()
    center_y = torch.randint(height, (1,), device=images.device).item()

    x1 = max(center_x - cut_w // 2, 0)
    x2 = min(center_x + cut_w // 2, width)
    y1 = max(center_y - cut_h // 2, 0)
    y2 = min(center_y + cut_h // 2, height)

    # replace the rectangle
    images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
    # set the lambda to the true replacement area ratio
    lam = 1.0 - ((x2 - x1) * (y2 - y1) / (width * height))
    return images, labels, labels[index], lam


def clone_state_dict_to_cpu(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in state_dict.items()
    }


def train(model: CustomModel, dataset: TinyImageNetDataset) -> None:
    train_transform = Compose(
        [
            RandomCrop(64, padding=8),
            RandomHorizontalFlip(),
            ToDtype(torch.float32, scale=True),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset.transform = train_transform

    batch_size = 128
    num_workers = min(8, os.cpu_count() or 1)
    ema_decay = 0.999
    cutmix_alpha = 0.2
    # pin-memory means page_locked memory, making data transfer from CPU to GPU faster
    pin_memory = (DEVICE.type == "cuda") 
    # automatic mixed precision
    use_amp = (DEVICE.type == "cuda")
    # tels pytorch to automatically choose lower precision for safe operations,
    # but keep sensitive operations to higher precision
    autocast_device = "cuda" if use_amp else "cpu"

    if use_amp:
        torch.backends.cudnn.benchmark = True # tell cuda to try different conv algorithms and choose the fastest one

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True, # workers in dataloader stay alive from epoch to epoch, not recreated
        prefetch_factor=4, # num of batches each worker prepares in advance
    )

    epochs = 50
    lr = 0.08
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=lr, 
        momentum=0.9, 
        weight_decay=5e-4,
        nesterov=True, # using nesterov momentum
    )
    criterion = torch.nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda",enabled=use_amp)
    ema = ModelEMA(model, decay=ema_decay)

    steps_per_epoch = len(train_loader)
    warmup_epochs = 2
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = epochs * steps_per_epoch
    # a scheduler changes the learning rate during training
    # high lr is useful early, lower lr is useful later for refinement
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    # cosine annealing is smooth decay, it often works well for img classification
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_steps - warmup_steps),
        eta_min = 1e-4,
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        running_samples = 0

        for images, labels in train_loader:
            images = images.to(DEVICE, non_blocking=pin_memory)
            labels = labels.to(DEVICE, non_blocking=pin_memory)

            images, labels_a, labels_b, lam = cutmix_batch(images, labels, cutmix_alpha)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(autocast_device, enabled=use_amp):
                logits = model(images)
                # uses mixed labels loss
                loss = lam * criterion(logits, labels_a) + (1.0 - lam) * criterion(logits, labels_b)
            
            scaler.scale(loss).backward() # scale it to avoid zero gradient
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)
            scheduler.step()

            preds = logits.argmax(dim=1)
            running_loss += loss.item() * labels.size(0)
            # use mixed label correctness
            running_corrects += (
                lam * (preds == labels_a).sum().item()
                + (1.0 - lam) * (preds == labels_b).sum().item()
            )
            running_samples += labels.size(0)

        train_loss = running_loss / running_samples
        train_acc = running_corrects / running_samples
        
        current_lr = optimizer.param_groups[0]["lr"]
        logging.info(
            f"Epoch {epoch + 1:03d}/{epochs} | "
            f"lr={current_lr:.5f} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_acc:.4f}"
        )

    ema.copy_to(model)


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = ArgumentParser()
    parser.add_argument("ckpt_path", type=Path)
    args = parser.parse_args()

    ckpt_path: Path = args.ckpt_path

    dataset = TinyImageNetDataset(DATASET_ROOT, "train", transform=TRANSFORM)

    model = CustomModel().to(DEVICE)

    if sum(p.numel() for p in model.parameters()) > 20_000_000:
        logging.error("Model has more than 20 million parameters")
        return

    train(model, dataset)

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)

    logging.info(f"Model checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    main()
