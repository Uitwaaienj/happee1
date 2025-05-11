import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from typing import List, Tuple
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split

class TEDataset(Dataset):
    def __init__(self, data_dir: str, window_size: int = 16, stride: int = 1, scaler: StandardScaler = None, is_train: bool = True):
        self.data_dir = data_dir
        self.window_size = window_size
        self.stride = stride
        self.scaler = scaler
        self.is_train = is_train
        self.data, self.labels = self._load_data()
        
    def _load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        all_data = []
        all_labels = []
        raw_data = []
        file_labels = []
        
        # 首先收集所有数据
        for file_name in sorted(os.listdir(self.data_dir)):
            if file_name.endswith('.dat'):
                file_path = os.path.join(self.data_dir, file_name)
                data = np.loadtxt(file_path)
                
                # 使用Savitzky-Golay滤波器平滑数据
                data = savgol_filter(data, window_length=7, polyorder=3, axis=0)
                
                # 从文件名中提取标签
                if '_te' in file_name:
                    label = int(file_name.split('_')[0][1:])
                else:
                    label = int(file_name[1:-4])
                
                raw_data.append(data)
                file_labels.append(label)
        
        # 将所有数据展平并标准化
        if self.scaler is None:
            self.scaler = StandardScaler()
            flat_data = np.vstack(raw_data)
            self.scaler.fit(flat_data)
        
        # 处理每个文件
        for data, label in zip(raw_data, file_labels):
            # 标准化数据
            data = self.scaler.transform(data)
            
            # 添加统计特征
            mean = np.mean(data, axis=0, keepdims=True)
            std = np.std(data, axis=0, keepdims=True)
            data = np.concatenate([data, mean.repeat(len(data), axis=0), std.repeat(len(data), axis=0)], axis=1)
            
            # 创建重叠的滑动窗口
            overlap = self.window_size // 2 if self.is_train else self.stride
            for i in range(0, len(data) - self.window_size + 1, overlap):
                window = data[i:i + self.window_size]
                
                # 添加窗口级别的统计特征
                window_mean = np.mean(window, axis=0, keepdims=True)
                window_std = np.std(window, axis=0, keepdims=True)
                window_max = np.max(window, axis=0, keepdims=True)
                window_min = np.min(window, axis=0, keepdims=True)
                
                # 组合所有特征
                window_features = np.concatenate([
                    window,
                    window_mean.repeat(self.window_size, axis=0),
                    window_std.repeat(self.window_size, axis=0),
                    window_max.repeat(self.window_size, axis=0),
                    window_min.repeat(self.window_size, axis=0)
                ], axis=1)
                
                all_data.append(window_features)
                all_labels.append(label)
        
        # 转换为张量
        data_tensor = torch.FloatTensor(all_data)
        labels_tensor = torch.LongTensor(all_labels)
        
        # 对训练数据进行轻微扰动
        if self.is_train:

            noise = torch.randn_like(data_tensor) * 0.001
            data_tensor = data_tensor + noise
        
        return data_tensor, labels_tensor
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]

def create_data_loaders(
    train_dir: str,
    test_dir: str,
    batch_size: int = 32,
    window_size: int = 16,
    stride: int = 1,
    test_size: float = 0.7,
    random_state: int = 48
) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和测试数据加载器
    """
    # 合并训练集和测试集
    all_data = []
    all_labels = []
    raw_data = []
    file_labels = []
    
    # 处理训练集
    for file_name in sorted(os.listdir(train_dir)):
        if file_name.endswith('.dat'):
            file_path = os.path.join(train_dir, file_name)
            data = np.loadtxt(file_path)
            label = int(file_name[1:-4])
            raw_data.append(data)
            file_labels.append(label)
    
    # 处理测试集
    for file_name in sorted(os.listdir(test_dir)):
        if file_name.endswith('.dat'):
            file_path = os.path.join(test_dir, file_name)
            data = np.loadtxt(file_path)
            label = int(file_name.split('_')[0][1:])
            raw_data.append(data)
            file_labels.append(label)
    
    # 创建scaler并标准化数据
    scaler = StandardScaler()
    flat_data = np.vstack(raw_data)
    scaler.fit(flat_data)
    
    # 处理每个文件
    for data, label in zip(raw_data, file_labels):
        # 标准化数据
        data = scaler.transform(data)
        
        # 添加统计特征
        mean = np.mean(data, axis=0, keepdims=True)
        std = np.std(data, axis=0, keepdims=True)
        data = np.concatenate([data, mean.repeat(len(data), axis=0), std.repeat(len(data), axis=0)], axis=1)
        
        # 创建滑动窗口
        for i in range(0, len(data) - window_size + 1, stride):
            window = data[i:i + window_size]
            
            # 添加窗口级别的统计特征
            window_mean = np.mean(window, axis=0, keepdims=True)
            window_std = np.std(window, axis=0, keepdims=True)
            window_max = np.max(window, axis=0, keepdims=True)
            window_min = np.min(window, axis=0, keepdims=True)
            
            # 组合所有特征
            window_features = np.concatenate([
                window,
                window_mean.repeat(window_size, axis=0),
                window_std.repeat(window_size, axis=0),
                window_max.repeat(window_size, axis=0),
                window_min.repeat(window_size, axis=0)
            ], axis=1)
            
            all_data.append(window_features)
            all_labels.append(label)
    
    # 转换为numpy数组
    all_data = np.array(all_data)
    all_labels = np.array(all_labels)
    
    # 随机划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        all_data, all_labels, 
        test_size=test_size, 
        random_state=random_state,
        stratify=all_labels
    )
    
    # 转换为张量
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)
    
    # 创建数据集
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )
    
    return train_loader, test_loader 