# dataset.py
"""
Dataset classes and data loading utilities for golf swing classification
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from config import TRAIN_CONFIG, WEIGHTED_SAMPLING_CONFIG


class GolfSwingDataset(Dataset):
    """
    Custom dataset class for golf swing pose sequences.
    
    Args:
        data_dir (str): Directory containing the processed data files
        swing_ids (list): List of swing IDs to include in this dataset
    """
    
    def __init__(self, data_dir, swing_ids):
        self.data_dir = data_dir
        self.swing_ids = swing_ids
        
    def __len__(self):
        return len(self.swing_ids)
    
    def __getitem__(self, idx):
        swing_id = self.swing_ids[idx]
        seq_path = os.path.join(self.data_dir, swing_id + "_seq.pt")
        label_path = os.path.join(self.data_dir, swing_id + "_label.pt")

        seq = torch.load(seq_path)      # [C, T, V] - Channels, Time, Vertices
        label = torch.load(label_path)  # int scalar

        return seq, label


def extract_swing_ids(data_root_dir):
    """
    Extract unique swing IDs from the data directory.
    
    Args:
        data_root_dir (str): Root directory containing data files
        
    Returns:
        list: Sorted list of unique swing IDs
    """
    all_files = os.listdir(data_root_dir)
    swing_ids = sorted(list(set([
        fname.rsplit("_", 1)[0].replace("_label", "").replace("_seq", "")
        for fname in all_files
        if fname.endswith(".pt") and ("_label" in fname or "_seq" in fname)
    ])))
    return swing_ids


def split_data(swing_ids, val_size=0.15, test_size=0.15, seed=42):
    """
    Split swing IDs into train, validation, and test sets.
    
    Args:
        swing_ids (list): List of swing IDs
        val_size (float): Proportion of data for validation
        test_size (float): Proportion of data for testing
        seed (int): Random seed for reproducibility
        
    Returns:
        tuple: (train_ids, val_ids, test_ids)
    """
    random.seed(seed)
    random.shuffle(swing_ids)

    total = len(swing_ids)
    test_split = int(test_size * total)
    val_split = int(val_size * total)
    train_split = total - test_split - val_split

    train_ids = swing_ids[:train_split]
    val_ids = swing_ids[train_split:train_split + val_split]
    test_ids = swing_ids[train_split + val_split:]

    return train_ids, val_ids, test_ids


def get_data_auto_split(data_root_dir, val_size=0.15, test_size=0.15, batch_size=8):
    """
    Create data loaders with automatic train/val/test split.
    
    Args:
        data_root_dir (str): Root directory containing data files
        val_size (float): Proportion of data for validation
        test_size (float): Proportion of data for testing
        batch_size (int): Batch size for data loaders
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Extract swing IDs
    swing_ids = extract_swing_ids(data_root_dir)
    
    # Split data
    train_ids, val_ids, test_ids = split_data(swing_ids, val_size, test_size)
    
    print(f"총 샘플 수: {len(swing_ids)}")
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

    # Create datasets
    train_dataset = GolfSwingDataset(data_root_dir, train_ids)
    val_dataset = GolfSwingDataset(data_root_dir, val_ids)
    test_dataset = GolfSwingDataset(data_root_dir, test_ids)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def get_data_auto_split_weighted(data_root_dir, val_size=0.15, test_size=0.15, batch_size=8):
    """
    Create data loaders with weighted sampling to handle class imbalance.
    
    Args:
        data_root_dir (str): Root directory containing data files
        val_size (float): Proportion of data for validation
        test_size (float): Proportion of data for testing
        batch_size (int): Batch size for data loaders
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Extract swing IDs and split data
    swing_ids = extract_swing_ids(data_root_dir)
    train_ids, val_ids, test_ids = split_data(swing_ids, val_size, test_size)

    print(f"총 샘플 수: {len(swing_ids)}")
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

    # Create datasets
    train_dataset = GolfSwingDataset(data_root_dir, train_ids)
    val_dataset = GolfSwingDataset(data_root_dir, val_ids)
    test_dataset = GolfSwingDataset(data_root_dir, test_ids)

    # Extract labels for weighted sampling
    labels = np.array([train_dataset[i][1].item() for i in range(len(train_dataset))])
    class_sample_count = np.bincount(labels)
    print(f"Train 클래스별 샘플 수: {class_sample_count}")

    # Calculate class weights (inversely proportional to sample count)
    num_classes = len(class_sample_count)
    total_samples = len(train_dataset)
    raw_weights = total_samples / (num_classes * class_sample_count)

    # Apply weight smoothing to prevent extreme weights
    weights_per_class = np.clip(
        raw_weights, 
        a_min=WEIGHTED_SAMPLING_CONFIG['weight_clip_min'], 
        a_max=WEIGHTED_SAMPLING_CONFIG['weight_clip_max']
    )

    print(f"스무딩된 클래스별 가중치: {weights_per_class}")

    # Create sample weights
    sample_weights = weights_per_class[labels].astype(np.float32)

    # Create weighted sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=WEIGHTED_SAMPLING_CONFIG['replacement']
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader