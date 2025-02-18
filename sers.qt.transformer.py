# -*- coding: utf-8 -*-

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.nn import functional as F
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from torch.nn.utils.rnn import pad_sequence

# 自定义数据集类
class RamanSpectraDataset(Dataset):
    def __init__(self, csv_file, input_dim):
        self.data = pd.read_csv(csv_file)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.input_dim = input_dim
        
        # 特征数据，拉曼光谱
        self.features = self.scaler.fit_transform(self.data.iloc[:, :-1].values)
        # 标签数据，分类标记
        self.labels = self.label_encoder.fit_transform(self.data.iloc[:, -1].values)
        
        # 检查每个光谱的长度是否等于input_dim
        if not all(len(feature) == input_dim for feature in self.features):
            raise ValueError("所有光谱的长度必须等于input_dim")

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx], dtype=torch.float)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return feature, label

# 定义Transformer模型
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, num_classes):
        super(TransformerClassifier, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # 线性嵌入层将input_dim转换为d_model
        self.embedding = nn.Linear(input_dim, d_model)
        
        # Transformer编码器
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True),
            num_layers=num_layers
        )
        
        # 全连接层用于分类
        self.fc = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        # 线性变换
        x = self.embedding(x)
        # Transformer编码
        x = self.transformer_encoder(x)
        # ReLU激活函数
        x = F.relu(x)
        # 全连接层
        x = self.fc(x)
        return x

# 参数定义
input_dim = 1015  # 原始拉曼光谱的长度
d_model = 64    # Transformer模型中每个序列元素的维度
nhead = 8       # d_model 必须能够整除 nhead
num_layers = 2  # Transformer层数，好像很关键，太多反而不如太少
num_classes = 3 # 假设有3个分类，根据实际情况调整
batch_size = 32
num_epochs = 25
learning_rate = 0.0001

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据集和加载器
dataset = RamanSpectraDataset('./results.csv', input_dim)  # 替换为你的CSV文件路径
train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = train_test_split(dataset, train_size=train_size, test_size=test_size, random_state=41)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 模型、损失函数、优化器定义
model = TransformerClassifier(input_dim, d_model, nhead, num_layers, num_classes).to(device)  # 将模型移动到设备
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    loss_history = []
    accuracy_history = []
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # 将输入数据和标签移动到设备
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            # 计算训练精度
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')
        
        # 记录损失和精度历史
        loss_history.append(epoch_loss)
        accuracy_history.append(epoch_accuracy)
    
    # 训练结束后保存模型
    torch.save(model.state_dict(), 'raman_transformer_model.pth')
    
    # 保存损失和精度历史到CSV文件
    history_df = pd.DataFrame({
        'Epoch': range(1, num_epochs + 1),
        'Loss': loss_history,
        'Accuracy': accuracy_history
    })
    history_df.to_csv('training_history.csv', index=False)
    
    # 可视化损失和精度
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history_df['Epoch'], history_df['Loss'], label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history_df['Epoch'], history_df['Accuracy'], label='Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy over Epochs')
    plt.legend()
    
    # 保存图片
    plt.savefig('training_history.png')
    plt.show()

# 加载模型
def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))  # 确保模型加载到正确设备
    model.eval()

# 测试模型并生成混淆矩阵
def test_model(model, test_loader, dataset):
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # 将输入数据和标签移动到设备
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 收集所有标签和预测结果
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    test_accuracy = 100 * correct / total
    print(f'Test Accuracy: {test_accuracy:.2f}%')
    
    # 生成混淆矩阵
    cm = confusion_matrix(all_labels, all_predictions)
    
    # 打印混淆矩阵
    print("Confusion Matrix:")
    print(cm)
    
    # 保存混淆矩阵到CSV文件
    cm_df = pd.DataFrame(cm, index=dataset.label_encoder.classes_, columns=dataset.label_encoder.classes_)
    cm_df.to_csv('confusion_matrix.csv', index=True)
    
    # 可视化混淆矩阵
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=dataset.label_encoder.classes_, yticklabels=dataset.label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # 保存图片
    plt.savefig('confusion_matrix.png')
    plt.show()

# 开始训练和测试
train_model(model, train_loader, criterion, optimizer, num_epochs)

# 加载保存的模型
loaded_model = TransformerClassifier(input_dim, d_model, nhead, num_layers, num_classes).to(device)  # 创建一个新的模型实例
load_model(loaded_model, 'raman_transformer_model.pth')  # 加载保存的权重

# 使用加载的模型进行测试并生成混淆矩阵
test_model(loaded_model, test_loader, dataset)
