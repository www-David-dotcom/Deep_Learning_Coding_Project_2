from pathlib import Path
from typing import Literal

import torchvision.io
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode
from torchvision.transforms.v2 import Transform


class TinyImageNetDataset(Dataset[tuple[Tensor, int]]):
    root: Path
    split: Literal["train", "val"]
    transform: Transform | None

    classes: list[str]

    img_paths: list[Path]
    labels: list[int]

    def __init__(
        self,
        root: str | Path,
        split: Literal["train", "val"],
        transform: Transform | None = None,
    ) -> None:
        super().__init__()

        self.root = Path(root)
        self.split = split
        self.transform = transform

        self.classes = (self.root / "wnids.txt").read_text().splitlines()

        match self.split:
            case "train":
                self.img_paths, self.labels = self._load_train()
            case "val":
                self.img_paths, self.labels = self._load_val()

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        img_path = self.img_paths[idx]
        image = torchvision.io.decode_image(str(img_path), mode=ImageReadMode.RGB)

        if self.transform is not None:
            image = self.transform(image)

        label = self.labels[idx]

        return image, label

    def _load_train(self) -> tuple[list[Path], list[int]]:
        img_paths: list[Path] = []
        labels: list[int] = []

        for class_idx, class_name in enumerate(self.classes):
            for image_path in (self.root / "train" / class_name / "images").glob(
                "*.JPEG"
            ):
                img_paths.append(image_path)
                labels.append(class_idx)

        return img_paths, labels

    def _load_val(self) -> tuple[list[Path], list[int]]:
        img_paths: list[Path] = []
        labels: list[int] = []

        val_annotations = (
            (self.root / "val" / "val_annotations.txt").read_text().splitlines()
        )

        image_to_label = dict(line.split("\t")[:2] for line in val_annotations)

        class_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}

        for image_path in (self.root / "val" / "images").glob("*.JPEG"):
            if image_path.name.startswith("._"):
                continue

            label = class_to_idx[image_to_label[image_path.name]]

            img_paths.append(image_path)
            labels.append(label)

        return img_paths, labels
