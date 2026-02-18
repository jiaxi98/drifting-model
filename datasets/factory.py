"""Train/eval dataset builders."""

from pathlib import Path
from typing import Optional, Tuple

from torchvision import datasets as tv_datasets
from torchvision import transforms


def get_dataset(
    name: str,
    root: str,
    img_size: int,
    include_eval: bool = True,
) -> Tuple[object, Optional[object]]:
    """Create dataset objects for train/eval."""
    dataset_name = name.lower()

    if dataset_name == "mnist":
        mnist_root = Path(root) / "mnist"
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        train_dataset = tv_datasets.MNIST(
            str(mnist_root),
            train=True,
            download=True,
            transform=transform,
        )
        test_dataset = (
            tv_datasets.MNIST(str(mnist_root), train=False, download=True, transform=transform)
            if include_eval
            else None
        )
        return train_dataset, test_dataset

    if dataset_name == "cifar":
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        train_dataset = tv_datasets.CIFAR10(
            root,
            train=True,
            download=True,
            transform=train_transform,
        )
        test_dataset = (
            tv_datasets.CIFAR10(root, train=False, download=True, transform=eval_transform)
            if include_eval
            else None
        )
        return train_dataset, test_dataset

    if dataset_name == "imagenet":
        train_dir = Path(root) / "train"
        val_dir = Path(root) / "val"
        if not train_dir.exists():
            raise FileNotFoundError(
                f"ImageNet expects {train_dir}. Prepare ImageFolder-style train directory first."
            )

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        eval_transform = transforms.Compose([
            transforms.Resize(int(img_size * 256 / 224)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        train_dataset = tv_datasets.ImageFolder(str(train_dir), transform=train_transform)
        if include_eval:
            has_val_classes = val_dir.exists() and any(path.is_dir() for path in val_dir.iterdir())
            test_dataset = (
                tv_datasets.ImageFolder(str(val_dir), transform=eval_transform)
                if has_val_classes
                else train_dataset
            )
        else:
            test_dataset = None
        return train_dataset, test_dataset

    raise ValueError(f"Unsupported dataset: {name}")
