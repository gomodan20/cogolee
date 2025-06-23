# training_utils.py
"""
Training and evaluation utilities for golf swing classification
"""

import torch


def train(model, train_loader, optimizer, criterion, device):
    """
    Train the model for one epoch.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        optimizer: Optimizer for updating model parameters
        criterion: Loss function
        device: Device to run training on (CPU/GPU)
        
    Returns:
        float: Average training loss for the epoch
    """
    model.train()
    total_loss = 0
    
    for data, label in train_loader:
        data = data.to(device)
        label = label.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        
        loss.backward()
        # loss.backward()는 PyTorch의 자동 미분(Autograd) 기능으로 
        # 계산 그래프를 따라가면서 모델 파라미터들에 대한 손실 함수의 기울기를 자동으로 계산
        optimizer.step()
        # optimizer.step()이 계산된 기울기를 이용해 파라미터를 업데이트
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def evaluate(model, data_loader, criterion, device):
    """
    Evaluate the model on validation or test data.
    
    Args:
        model: Neural network model
        data_loader: Data loader (validation or test)
        criterion: Loss function
        device: Device to run evaluation on (CPU/GPU)
        
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, label in data_loader:
            data = data.to(device)
            label = label.to(device)
            
            output = model(data)
            loss = criterion(output, label)
            
            total_loss += loss.item()
            
            _, pred = output.max(1)
            correct += (pred == label).sum().item()
            total += label.size(0)
    
    accuracy = correct / total
    return total_loss / len(data_loader), accuracy


def get_predictions(model, data_loader, device):
    """
    Get model predictions for a dataset.
    
    Args:
        model: Neural network model
        data_loader: Data loader
        device: Device to run inference on (CPU/GPU)
        
    Returns:
        tuple: (predictions, true_labels)
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in data_loader:
            data, labels = data.to(device), labels.to(device)
            preds = torch.argmax(model(data), dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return all_preds, all_labels