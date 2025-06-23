# train.py
"""
Main training script for golf swing classification
"""

import os
import csv
import torch
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import MODEL_CONFIG, TRAIN_CONFIG, FOCAL_LOSS_CONFIG, SCHEDULER_CONFIG
from dataset import get_data_auto_split, get_data_auto_split_weighted
from loss_functions import SparseCategoricalFocalLoss
from training_utils import train, evaluate
from visualization import evaluate_and_visualize, plot_training_history
from STGCN import Model  # Assuming STGCN model is in a separate file


def train_model(data_root_dir, model_save_dir, project_name, weighted=True, 
                val_size=0.15, test_size=0.15, batch_size=32, num_class=3, 
                num_epochs=200, early_stop_patience=20):
    """
    Complete training pipeline for golf swing classification.
    
    Args:
        data_root_dir (str): Root directory containing processed data
        model_save_dir (str): Directory to save trained models
        project_name (str): Name of the training project
        weighted (bool): Whether to use weighted sampling for class imbalance
        val_size (float): Proportion of data for validation
        test_size (float): Proportion of data for testing
        batch_size (int): Batch size for training
        num_class (int): Number of classes
        num_epochs (int): Maximum number of training epochs
        early_stop_patience (int): Early stopping patience
        
    Returns:
        tuple: Training results and predictions
    """
    
    # Load data
    print("🔄 데이터 로딩 중...")
    if weighted:
        train_loader, val_loader, test_loader = get_data_auto_split_weighted(
            data_root_dir, val_size, test_size, batch_size
        )
    else:
        train_loader, val_loader, test_loader = get_data_auto_split(
            data_root_dir, val_size, test_size, batch_size
        )

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ 사용 디바이스: {device}")

    # Create save directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_dir = os.path.join(model_save_dir, f"{project_name}_{timestamp}")
    log_dir = os.path.join(project_dir, 'logs')
    os.makedirs(project_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Define save paths
    model_acc_path = os.path.join(project_dir, "best_val_acc_model.pt")
    model_loss_path = os.path.join(project_dir, "best_val_loss_model.pt")
    log_csv_path = os.path.join(log_dir, "train_log.csv")

    # Initialize model
    print("🔄 모델 초기화 중...")
    model_config = MODEL_CONFIG.copy()
    model_config['num_class'] = num_class
    
    model = Model(
        in_channels=model_config['in_channels'],
        num_class=model_config['num_class'],
        edge_importance_weighting=model_config['edge_importance_weighting'],
        graph_args=model_config['graph_args'],
        dropout=model_config['dropout']
    ).to(device)

    # Initialize optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_CONFIG['learning_rate'])
    criterion = SparseCategoricalFocalLoss(
        gamma=FOCAL_LOSS_CONFIG['gamma'],
        weight=FOCAL_LOSS_CONFIG['weight'],
        reduction=FOCAL_LOSS_CONFIG['reduction']
    )
    scheduler = ReduceLROnPlateau(optimizer, **SCHEDULER_CONFIG)

    # Training variables
    best_val_acc = 0.0
    best_val_loss = float('inf')
    early_stop_counter = 0

    # Initialize training log
    with open(log_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_acc'])

    print("🚀 훈련 시작!")
    print(f"📊 프로젝트: {project_name}")
    print(f"📁 저장 경로: {project_dir}")
    
    # Training loop
    for epoch in range(num_epochs):
        # Train for one epoch
        train_loss = train(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1:3d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}")

        # Log training progress
        with open(log_csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, train_loss, val_loss, val_acc])

        # Save best model based on accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_acc_path)
            print(f"✅ [ACC] 모델 저장됨 (Epoch {epoch+1}, Val Acc: {val_acc:.4f})")

        # Save best model based on loss and check early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_loss_path)
            print(f"✅ [LOSS] 모델 저장됨 (Epoch {epoch+1}, Val Loss: {val_loss:.4f})")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_patience:
                print(f"🛑 Early stopping triggered at epoch {epoch+1}")
                break

        # Show early stopping progress
        if early_stop_counter > 0:
            print(f"⏳ EarlyStopping 카운터: {early_stop_counter}/{early_stop_patience}")

    print("\n🎉 훈련 완료!")
    
    # Plot training history
    plot_training_history(log_csv_path, project_dir, timestamp)

    # Evaluate both models on test set
    print("\n📊 테스트 세트 평가 중...")
    
    preds_acc, labels_acc = evaluate_and_visualize(
        model, model_acc_path, test_loader, device, num_class,
        "Best Val Acc", project_dir, timestamp
    )
    
    preds_loss, labels_loss = evaluate_and_visualize(
        model, model_loss_path, test_loader, device, num_class,
        "Best Val Loss", project_dir, timestamp
    )

    print(f"\n🎯 최종 결과:")
    print(f"📁 모델 저장 경로: {project_dir}")
    print(f"📊 Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"📉 Best Validation Loss: {best_val_loss:.4f}")

    return {
        'preds_acc': preds_acc,
        'labels_acc': labels_acc,
        'preds_loss': preds_loss,
        'labels_loss': labels_loss,
        'best_val_acc': best_val_acc,
        'best_val_loss': best_val_loss,
        'project_dir': project_dir
    }


def main():
    """
    Main function to run training with default parameters.
    """
    # Default paths - modify these according to your setup
    data_root_dir = "D:/STGCN/data/processed_data_last"
    model_save_dir = "D:/STGCN/model"
    
    # Training configuration
    config = {
        'data_root_dir': data_root_dir,
        'model_save_dir': model_save_dir,
        'project_name': 'golf_swing_classification',
        'weighted': True,
        'val_size': TRAIN_CONFIG['val_size'],
        'test_size': TRAIN_CONFIG['test_size'],
        'batch_size': TRAIN_CONFIG['batch_size'],
        'num_class': MODEL_CONFIG['num_class'],
        'num_epochs': TRAIN_CONFIG['num_epochs'],
        'early_stop_patience': TRAIN_CONFIG['early_stop_patience']
    }
    
    print("🏌️ Golf Swing Classification Training")
    print("=" * 50)
    print(f"Data Directory: {config['data_root_dir']}")
    print(f"Model Save Directory: {config['model_save_dir']}")
    print(f"Weighted Sampling: {config['weighted']}")
    print(f"Batch Size: {config['batch_size']}")
    print(f"Max Epochs: {config['num_epochs']}")
    print("=" * 50)
    
    # Start training
    results = train_model(**config)
    
    print("\n✅ 훈련이 성공적으로 완료되었습니다!")
    return results


if __name__ == "__main__":
    main()
