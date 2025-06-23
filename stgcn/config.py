# config.py
"""
Configuration file for golf swing classification project
"""

# Data configuration
KEYPOINTS = [
    "head",         # 0
    "neck",         # 1
    "chest",        # 2
    "right_shoulder", # 3
    "left_shoulder",  # 4
    "right_elbow",  # 5
    "left_elbow",   # 6
    "right_wrist",  # 7
    "left_wrist",   # 8
    "hip",          # 9 (center hip)
    "right_hip",    # 10
    "left_hip",     # 11
    "right_knee",   # 12
    "left_knee",    # 13
    "right_ankle",  # 14
    "left_ankle"    # 15
]

# Model configuration
MODEL_CONFIG = {
    'in_channels': 2,
    'num_class': 3,
    'edge_importance_weighting': True,
    'graph_args': {'layout': 'golf', 'strategy': 'spatial'},
    'dropout': 0.3
}

# Training configuration
TRAIN_CONFIG = {
    'batch_size': 32,
    'num_epochs': 200,
    'learning_rate': 0.0005,
    'early_stop_patience': 20,
    'val_size': 0.15,
    'test_size': 0.15,
    'random_seed': 42
}

# Focal loss configuration
FOCAL_LOSS_CONFIG = {
    'gamma': 2.0,
    'weight': None,
    'reduction': 'mean'
}

# Weighted sampling configuration
WEIGHTED_SAMPLING_CONFIG = {
    'weight_clip_min': 0.5,
    'weight_clip_max': 5.0,
    'replacement': True
}

# Scheduler configuration
SCHEDULER_CONFIG = {
    'mode': 'min',
    'factor': 0.5,
    'patience': 5,
    'verbose': True
}

# Default paths (can be overridden)
DEFAULT_PATHS = {
    'data_root_dir': "D:/STGCN/data/processed_data_last",
    'model_save_dir': "D:/STGCN/model"
}