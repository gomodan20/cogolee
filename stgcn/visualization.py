# visualization.py
"""
Visualization utilities for model evaluation
"""

import os
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import torch

from training_utils import get_predictions


def plot_confusion_matrix(all_labels, all_preds, num_classes, title, save_path):
    """
    Plot and save confusion matrix.
    
    Args:
        all_labels (list): True labels
        all_preds (list): Predicted labels
        num_classes (int): Number of classes
        title (str): Title for the plot
        save_path (str): Path to save the plot
    """
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f"Class {i}" for i in range(num_classes)],
                yticklabels=[f"Class {i}" for i in range(num_classes)])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{title} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"📁 Confusion matrix 저장됨: {save_path}")


def save_predictions_csv(all_labels, all_preds, save_path):
    """
    Save predictions to CSV file.
    
    Args:
        all_labels (list): True labels
        all_preds (list): Predicted labels
        save_path (str): Path to save the CSV file
    """
    with open(save_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['index', 'true_label', 'predicted_label'])
        for i, (true, pred) in enumerate(zip(all_labels, all_preds)):
            writer.writerow([i, true, pred])
    print(f"📁 예측 결과 CSV 저장 완료: {save_path}")


def evaluate_and_visualize(model, model_path, test_loader, device, num_classes, 
                          title, save_dir, timestamp):
    """
    Load model, evaluate on test data, and create visualizations.
    
    Args:
        model: Neural network model
        model_path (str): Path to saved model weights
        test_loader: Test data loader
        device: Device to run evaluation on
        num_classes (int): Number of classes
        title (str): Title for plots and reports
        save_dir (str): Directory to save outputs
        timestamp (str): Timestamp for file naming
        
    Returns:
        tuple: (predictions, true_labels)
    """
    # Load model weights
    model.load_state_dict(torch.load(model_path))
    
    # Get predictions
    all_preds, all_labels = get_predictions(model, test_loader, device)
    
    # Save predictions to CSV
    csv_path = os.path.join(save_dir, f'test_predictions_{title.lower().replace(" ", "_")}.csv')
    save_predictions_csv(all_labels, all_preds, csv_path)
    
    # Print confusion matrix and classification report
    cm = confusion_matrix(all_labels, all_preds)
    print(f"\n📊 Confusion Matrix ({title}):")
    print(cm)
    
    print(f"\n📋 Classification Report ({title}):")
    print(classification_report(all_labels, all_preds, 
                              target_names=[f"Class {i}" for i in range(num_classes)]))
    
    # Plot and save confusion matrix
    plot_name = f"confusion_matrix_{title.lower().replace(' ', '_')}_{timestamp}.png"
    plot_path = os.path.join(save_dir, plot_name)
    plot_confusion_matrix(all_labels, all_preds, num_classes, title, plot_path)
    
    return all_preds, all_labels


def plot_training_history(log_csv_path, save_dir, timestamp):
    """
    Plot training history from log CSV file.
    
    Args:
        log_csv_path (str): Path to training log CSV file
        save_dir (str): Directory to save the plot
        timestamp (str): Timestamp for file naming
    """
    try:
        import pandas as pd
        
        # Read training log
        df = pd.read_csv(log_csv_path)
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot losses
        ax1.plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o')
        ax1.plot(df['epoch'], df['val_loss'], label='Validation Loss', marker='s')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(df['epoch'], df['val_acc'], label='Validation Accuracy', marker='^', color='green')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(save_dir, f'training_history_{timestamp}.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"📁 Training history plot 저장됨: {plot_path}")
        
    except ImportError:
        print("⚠️ pandas가 설치되지 않아 training history plot을 생성할 수 없습니다.")
    except Exception as e:
        print(f"⚠️ Training history plot 생성 중 오류 발생: {e}")