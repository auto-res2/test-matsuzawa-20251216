import torch
from torch.utils.data import DataLoader, random_split, Dataset
import torchvision
from torchvision import datasets, transforms
import numpy as np
from typing import Dict, Any, Tuple
from pathlib import Path


class SyntheticSequenceDataset(Dataset):
    """Synthetic sequence dataset for Penn Treebank simulation"""
    
    def __init__(self, num_samples: int, seq_length: int, embedding_dim: int, num_classes: int):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        
        self.data = []
        np.random.seed(42)
        for _ in range(num_samples):
            embedding = torch.randn(seq_length, embedding_dim, dtype=torch.float32)
            label = torch.randint(0, num_classes, (1,)).item()
            self.data.append((embedding, label))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        embedding, label = self.data[idx]
        return embedding, label


def get_data_loaders(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load and prepare data loaders based on config"""
    
    dataset_name = cfg["dataset"].get("name", "CIFAR-10").lower()
    batch_size = cfg["training"]["batch_size"]
    
    if "cifar-10" in dataset_name:
        return _get_cifar10_loaders(cfg, batch_size)
    elif "cifar-100" in dataset_name:
        return _get_cifar100_loaders(cfg, batch_size)
    elif "penn" in dataset_name or "treebank" in dataset_name:
        return _get_penntreebank_loaders(cfg, batch_size)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def _get_cifar10_loaders(cfg: Dict[str, Any], batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """CIFAR-10 data loaders"""
    
    mean = cfg["dataset"]["normalization"]["mean"]
    std = cfg["dataset"]["normalization"]["std"]
    
    # Validate normalization stats
    assert len(mean) == 3, f"Expected 3 mean values for CIFAR-10, got {len(mean)}"
    assert len(std) == 3, f"Expected 3 std values for CIFAR-10, got {len(std)}"
    assert all(s > 0 for s in std), "Std values must be positive"
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    cache_dir = Path(".cache/cifar10")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    train_dataset = datasets.CIFAR10(
        root=str(cache_dir),
        train=True,
        download=True,
        transform=train_transform
    )
    
    test_dataset = datasets.CIFAR10(
        root=str(cache_dir),
        train=False,
        download=True,
        transform=test_transform
    )
    
    train_split = cfg["dataset"].get("train_split", 0.9)
    train_size = int(len(train_dataset) * train_split)
    val_size = len(train_dataset) - train_size
    train_set, val_set = random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def _get_cifar100_loaders(cfg: Dict[str, Any], batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """CIFAR-100 data loaders"""
    
    mean = cfg["dataset"]["normalization"]["mean"]
    std = cfg["dataset"]["normalization"]["std"]
    
    # Validate normalization stats
    assert len(mean) == 3, f"Expected 3 mean values for CIFAR-100, got {len(mean)}"
    assert len(std) == 3, f"Expected 3 std values for CIFAR-100, got {len(std)}"
    assert all(s > 0 for s in std), "Std values must be positive"
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    cache_dir = Path(".cache/cifar100")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    train_dataset = datasets.CIFAR100(
        root=str(cache_dir),
        train=True,
        download=True,
        transform=train_transform
    )
    
    test_dataset = datasets.CIFAR100(
        root=str(cache_dir),
        train=False,
        download=True,
        transform=test_transform
    )
    
    train_split = cfg["dataset"].get("train_split", 0.9)
    train_size = int(len(train_dataset) * train_split)
    val_size = len(train_dataset) - train_size
    train_set, val_set = random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def _get_penntreebank_loaders(cfg: Dict[str, Any], batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Penn Treebank data loaders (synthetic sequence data)"""
    
    seq_length = cfg["dataset"].get("sequence_length", 35)
    embedding_dim = cfg["dataset"].get("embedding_dim", 100)
    num_classes = cfg["model"].get("num_classes", 10)
    
    num_train = 5000
    num_val = 500
    num_test = 500
    
    train_dataset = SyntheticSequenceDataset(
        num_samples=num_train,
        seq_length=seq_length,
        embedding_dim=embedding_dim,
        num_classes=num_classes
    )
    
    val_dataset = SyntheticSequenceDataset(
        num_samples=num_val,
        seq_length=seq_length,
        embedding_dim=embedding_dim,
        num_classes=num_classes
    )
    
    test_dataset = SyntheticSequenceDataset(
        num_samples=num_test,
        seq_length=seq_length,
        embedding_dim=embedding_dim,
        num_classes=num_classes
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader
