import logging
from argparse import ArgumentParser
from pathlib import Path

import torch
from copy import deepcopy
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import (
    Compose,
    Normalize,
    RandomCrop,
    RandomErasing,
    RandomHorizontalFlip,
    ToDtype,
)

from datasets import TinyImageNetDataset
from evaluate import DATASET_ROOT, DEVICE, TRANSFORM, evaluate
from modules import CustomModel

try:
    import wandb
except ImportError:
    wandb = None



def train(model: CustomModel, dataset: TinyImageNetDataset) -> float:
    train_transform = Compose(
        [
            RandomCrop(64, padding=8),
            RandomHorizontalFlip(),
            ToDtype(torch.float32, scale=True),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            RandomErasing(p=0.25)
        ]
    )
    dataset.transform = train_transform
    val_dataset = TinyImageNetDataset(DATASET_ROOT, "val", transform=TRANSFORM)

    batch_size= 128
    pin_memory = (DEVICE.type == "cuda")
    use_amp = (DEVICE.type == "cuda")
    autocast_device = "cuda" if use_amp else "cpu"

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=pin_memory,
    )
    val_loader_batch_size = 128


    epochs = 60
    lr = 0.1 * batch_size / 256
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=lr, 
        momentum=0.9, 
        weight_decay=5e-4,
        nesterov=True,
    )
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler("cuda",enabled=use_amp)

    steps_per_epoch = len(train_loader)
    warmup_epochs = 5
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = epochs * steps_per_epoch
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
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

    wandb_run = None
    if wandb is not None:
        wandb_run = wandb.init(
            project="tiny-imagenet-resnext",
            config={
                "architecture": "Tiny-ResNeXt-50-16x4d",
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": lr,
                "optimizer": "SGD",
                "momentum": 0.9,
                "weight_decay": 5e-4,
                "label_smoothing": 0.1,
                "random_erasing_p": 0.25,
                "warmup_epochs": warmup_epochs,
                "num_parameters": sum(p.numel() for p in model.parameters()),
                "device": DEVICE.type,
            },
        )
    else:
        logging.warning("wandb is not installed; skipping WandB logging")

    best_val_acc = 0.0
    best_state_dict = deepcopy(model.state_dict())

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        running_samples = 0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(autocast_device, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            preds = logits.argmax(dim=1)
            running_loss += loss.item() * labels.size(0)
            running_corrects += (preds == labels).sum().item()
            running_samples += labels.size(0)

        train_loss = running_loss / running_samples
        train_acc = running_corrects / running_samples


        val_acc = evaluate(
            model,
            val_dataset,
            batch_size=val_loader_batch_size,
            num_workers=4,
            pin_memory=pin_memory,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = deepcopy(model.state_dict())
        
        current_lr = optimizer.param_groups[0]["lr"]
        logging.info(
            f"Epoch {epoch + 1:03d}/{epochs} | "
            f"lr={current_lr:.5f} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_acc:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"best_val_acc={best_val_acc:.4f}"
        )

        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch + 1,
                    "train/loss": train_loss,
                    "train/acc": train_acc,
                    "val/acc": val_acc,
                    "best_val_acc": best_val_acc,
                    "lr": current_lr,
                }
            )

    model.load_state_dict(best_state_dict)
    if wandb_run is not None:
        wandb_run.summary["best_val_acc"] = best_val_acc

    return best_val_acc







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

    try:
        best_val_acc = train(model, dataset)

        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)

        logging.info(f"Model checkpoint saved to {ckpt_path}")

        if wandb is not None and wandb.run is not None:
            wandb.run.summary["checkpoint_path"] = str(ckpt_path)
            wandb.run.summary["saved_best_val_acc"] = best_val_acc

            artifact = wandb.Artifact(
                "best-model",
                type="model",
                metadata={"best_val_acc": best_val_acc},
            )
            artifact.add_file(str(ckpt_path))
            wandb.log_artifact(artifact)
    finally:
        if wandb is not None and wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
