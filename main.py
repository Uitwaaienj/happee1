import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from data_processor import create_data_loaders
from model_factory import ModelFactory
import time
from datetime import datetime
import json
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import label_binarize
from itertools import cycle

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    start_time = time.time()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    
    train_time = time.time() - start_time
    return total_loss / len(train_loader), 100. * correct / total, train_time

def my_test(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    all_probs = []
    start_time = time.time()
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(torch.softmax(output, dim=1).cpu().numpy())
    
    test_time = time.time() - start_time
    return total_loss / len(test_loader), 100. * correct / total, all_preds, all_targets, all_probs, test_time

def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', 
                annot_kws={"size": 12}, cbar_kws={"shrink": .8})
    plt.title('Confusion Matrix (Normalized)', fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_loss_curves(train_losses, test_losses, save_dir, model_name):
    plt.figure(figsize=(12, 8))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(test_losses, label='Test Loss', linewidth=2)
    plt.title('Loss Curves', fontsize=16, pad=20)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_loss_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_accuracy_curves(train_accs, test_accs, save_dir, model_name):
    plt.figure(figsize=(12, 8))
    plt.plot(train_accs, label='Train Accuracy', linewidth=2)
    plt.plot(test_accs, label='Test Accuracy', linewidth=2)
    plt.title('Accuracy Curves', fontsize=16, pad=20)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_accuracy_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_curve(y_true, y_probs, num_classes, save_dir, model_name):
    # 将标签转换为one-hot编码
    y_true_bin = label_binarize(y_true, classes=range(num_classes))
    
    # 计算每个类别的ROC曲线
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], np.array(y_probs)[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # 绘制ROC曲线
    plt.figure(figsize=(12, 8))
    colors = cycle(['blue', 'red', 'green', 'yellow', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive'])
    
    for i, color in zip(range(num_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC Curves for Each Class', fontsize=16, pad=20)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_roc_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_evaluation_metrics(y_true, y_pred, save_dir, model_name):
    # 计算各项指标
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }
    
    # 绘制柱状图
    plt.figure(figsize=(12, 8))
    bars = plt.bar(metrics.keys(), metrics.values(), color='skyblue')
    plt.title('Evaluation Metrics', fontsize=16, pad=20)
    plt.ylim(0, 1)
    plt.ylabel('Score', fontsize=14)
    
    # 在柱子上添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=12)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_evaluation_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return metrics

def plot_class_distribution(y_true, y_pred, save_dir, model_name):
    # 计算真实标签和预测标签的分布
    true_counts = np.bincount(y_true)
    pred_counts = np.bincount(y_pred)
    
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 绘制真实标签分布
    ax1.bar(range(len(true_counts)), true_counts, color='skyblue')
    ax1.set_title('True Label Distribution', fontsize=14)
    ax1.set_xlabel('Class', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 绘制预测标签分布
    ax2.bar(range(len(pred_counts)), pred_counts, color='lightgreen')
    ax2.set_title('Predicted Label Distribution', fontsize=14)
    ax2.set_xlabel('Class', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    #plt.savefig(os.path.join(save_dir, f'{model_name}_class_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='TE Dataset Classification')
    parser.add_argument('--model_type', type=str, default='cnn_lstm',
                      choices=['cnn_gru', 'cnn_lstm', 'cnn', 'gru', 'lstm'],
                      help='Type of model to use')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--window_size', type=int, default=16)
    parser.add_argument('--stride', type=int, default=16)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--learning_rate', type=float, default=0.004)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--save_dir', type=str, default='./results',
                      help='Directory to save results')
    args = parser.parse_args()

    # 创建保存目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_save_dir = os.path.join(args.save_dir, f'{args.model_type}_{timestamp}')
    os.makedirs(model_save_dir, exist_ok=True)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 创建数据加载器
    train_loader, test_loader = create_data_loaders(
        train_dir='./训练集',
        test_dir='./测试集',
        batch_size=args.batch_size,
        window_size=args.window_size,
        stride=args.stride
    )

    # 获取输入大小和类别数
    sample_data, _ = next(iter(train_loader))
    input_size = sample_data.shape[2]  # (batch_size, seq_len, input_size)
    num_classes = len(torch.unique(torch.cat([y for _, y in train_loader])))

    # 创建模型
    model_factory = ModelFactory(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_classes=num_classes,
        dropout=args.dropout
    )
    model = model_factory.create_model(args.model_type).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # 训练循环
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    best_acc = 0
    best_model_state = None
    best_test_preds = None
    best_test_targets = None
    best_test_probs = None
    total_train_time = 0
    total_test_time = 0

    for epoch in range(args.num_epochs):
        start_time = time.time()
        
        train_loss, train_acc, train_time = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc, test_preds, test_targets, test_probs, test_time = my_test(model, test_loader, criterion, device)
        
        total_train_time += train_time
        total_test_time += test_time
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        epoch_time = time.time() - start_time
        
        print(f'Epoch {epoch+1}/{args.num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Train Time: {train_time:.2f}s')
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, Test Time: {test_time:.2f}s')
        print(f'Epoch Time: {epoch_time:.2f}s')
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            best_model_state = model.state_dict()
            best_test_preds = test_preds
            best_test_targets = test_targets
            best_test_probs = test_probs
            torch.save(best_model_state, os.path.join(model_save_dir, 'best_model.pth'))
    
    # 绘制并保存指标图
    plot_loss_curves(train_losses, test_losses, model_save_dir, args.model_type)
    plot_accuracy_curves(train_accs, test_accs, model_save_dir, args.model_type)
    plot_confusion_matrix(best_test_targets, best_test_preds, 
                         os.path.join(model_save_dir, f'{args.model_type}_confusion_matrix.png'))
    plot_roc_curve(best_test_targets, best_test_probs, num_classes, model_save_dir, args.model_type)
    plot_class_distribution(best_test_targets, best_test_preds, model_save_dir, args.model_type)
    metrics = plot_evaluation_metrics(best_test_targets, best_test_preds, model_save_dir, args.model_type)
    
    # 生成分类报告
    report = classification_report(best_test_targets, best_test_preds)
    with open(os.path.join(model_save_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
    
    # 保存训练配置和结果
    with open(os.path.join(model_save_dir, 'config.txt'), 'w') as f:
        f.write(f'Model Type: {args.model_type}\n')
        f.write(f'Batch Size: {args.batch_size}\n')
        f.write(f'Window Size: {args.window_size}\n')
        f.write(f'Hidden Size: {args.hidden_size}\n')
        f.write(f'Num Layers: {args.num_layers}\n')
        f.write(f'Dropout: {args.dropout}\n')
        f.write(f'Learning Rate: {args.learning_rate}\n')
        f.write(f'Num Epochs: {args.num_epochs}\n')
        f.write(f'Best Test Accuracy: {best_acc:.2f}%\n')
        f.write(f'\nTraining Time Statistics:\n')
        f.write(f'Total Training Time: {total_train_time:.2f}s\n')
        f.write(f'Average Training Time per Epoch: {total_train_time/args.num_epochs:.2f}s\n')
        f.write(f'Total Testing Time: {total_test_time:.2f}s\n')
        f.write(f'Average Testing Time per Epoch: {total_test_time/args.num_epochs:.2f}s\n')
        f.write(f'Total Time: {total_train_time + total_test_time:.2f}s\n')
        f.write('\nEvaluation Metrics:\n')
        for metric, value in metrics.items():
            f.write(f'{metric}: {value:.4f}\n')

    # 保存评估结果
    results = {
        'model_type': args.model_type,
        'hyperparameters': vars(args),
        'metrics': metrics,
        'time_statistics': {
            'total_training_time': total_train_time,
            'average_training_time_per_epoch': total_train_time/args.num_epochs,
            'total_testing_time': total_test_time,
            'average_testing_time_per_epoch': total_test_time/args.num_epochs,
            'total_time': total_train_time + total_test_time
        },
        'training_history': {
            'train_loss': train_losses,
            'test_loss': test_losses,
            'train_acc': train_accs,
            'test_acc': test_accs
        }
    }
    
    with open(os.path.join(model_save_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    main() 