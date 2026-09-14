import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class FlattenTransform:
    """Transforms a 2D image tensor to a 1D vector."""
    def __call__(self, x):
        # x is of shape (C, H, W)
        return x.view(-1)

def get_dataloaders(data_dir, batch_size=None, is_full_batch=False, seed=42, device=None):
    """
    Returns train, val, and test dataloaders.
    
    Args:
        data_dir (str): Path to the Group_11 dataset folder containing 'train', 'val', 'test'
        batch_size (int): Batch size to use (ignored if is_full_batch=True)
        is_full_batch (bool): If True, returns the entire dataset in a single batch
        seed (int): Seed for deterministic shuffling of the training DataLoader
        device (torch.device or None): Target device; when CUDA, enables pin_memory
    """
    transform = transforms.Compose([
        transforms.Grayscale(), # ensure it's 1 channel
        transforms.ToTensor(),
        FlattenTransform()
    ])

    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')

    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

    # Determine batch sizes
    train_bs = len(train_dataset) if is_full_batch else batch_size
    val_bs = len(val_dataset) # For validation/test we can always use full batch or a large batch
    test_bs = len(test_dataset)
    
    # If is_full_batch is true, we want the whole dataset. 
    # For optimizers like Adam or SGD, batch_size=1 is required by assignment.
    # We will pass batch_size=1 for them, and is_full_batch=True for BGD, AdaGrad, RMSProp.

    use_pin_memory = (device is not None and device.type == 'cuda')
    train_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_bs,
        shuffle=True,
        generator=train_generator,
        pin_memory=use_pin_memory,
        num_workers=2,
        prefetch_factor=2,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_bs,
        shuffle=False,
        pin_memory=use_pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_bs,
        shuffle=False,
        pin_memory=use_pin_memory,
    )

    return train_loader, val_loader, test_loader

def get_num_classes(data_dir):
    train_dir = os.path.join(data_dir, 'train')
    train_dataset = datasets.ImageFolder(root=train_dir)
    return len(train_dataset.classes)
