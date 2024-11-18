import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.optim as optim

# 1. 載入資料
data = np.load('data.npz')
X_train = data['X_train']
y_train = data['y_train']
X_test = data['X_test']

# 2. 預處理：將像素值正規化到 [0, 1]
X_train = X_train / 255.0  # 將數值從 0~255 正規化到 0~1
X_test = X_test / 255.0

# 將維度調整為 [Batch, Channels, Height, Width]
X_train = torch.tensor(X_train, dtype=torch.float32).permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32).permute(0, 3, 1, 2)

# 建立 TensorDataset
dataset = TensorDataset(X_train, y_train)

# 分割資料集為 training 和 validation
train_size = int(0.8 * len(dataset))  # 80% 作為訓練集
val_size = len(dataset) - train_size  # 剩餘 20% 作為驗證集
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 建立 DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 3. 定義 CNN 模型
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # 卷積層
        self.conv1 = nn.Conv2d(1, 2, kernel_size=3, padding=1)  # 1 -> 8
        self.conv2 = nn.Conv2d(2, 3, kernel_size=3, padding=1)  # 8 -> 16
        self.conv3 = nn.Conv2d(3, 4, kernel_size=3, padding=1)  # 16 -> 32
        self.pool = nn.MaxPool2d(2, 2)  # Max Pooling (每次尺寸減半)
        
        # 全連接層
        self.fc1 = nn.Linear(4 * 4 * 4, 128)  # 32 個通道，尺寸為 4x4
        self.fc2 = nn.Linear(128, 1)          # 輸出層 (二分類)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))  # Conv1 + ReLU
        x = self.pool(x)           # MaxPooling (32x32 -> 16x16)
        x = F.relu(self.conv2(x))  # Conv2 + ReLU
        x = self.pool(x)           # MaxPooling (16x16 -> 8x8)
        x = F.relu(self.conv3(x))  # Conv3 + ReLU
        x = self.pool(x)           # MaxPooling (8x8 -> 4x4)
        
        x = x.contiguous().view(-1, 4 * 4 * 4)  # 展平為全連接層的輸入
        x = F.relu(self.fc1(x))    # Fully Connected + ReLU
        x = torch.sigmoid(self.fc2(x))  # Fully Connected + Sigmoid
        return x

# 4. 定義模型、損失函數與優化器
model = CNNModel()
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# 5. 訓練模型
epochs = 100
for epoch in range(epochs):
    model.train()  # 設置為訓練模式
    train_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()  # 模型輸出展平
        loss = criterion(outputs, targets.squeeze())  # 壓縮目標形狀
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # 驗證模型
    model.eval()  # 設置為評估模式
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, targets.squeeze())
            val_loss += loss.item()
    
    # 計算平均損失
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}")

# 6. 評估並生成 CSV
model.eval()  # 設置為評估模式
with torch.no_grad():
    predictions = model(X_test).squeeze()
    predictions = (predictions > 0.5).float()  # 將輸出轉為二分類標籤

# 讀取樣板並生成提交檔案
sample_submission = pd.read_csv('Sample_submission.csv')
sample_submission['label'] = predictions.cpu().numpy().astype(int)  # 預測結果寫入樣板
sample_submission.to_csv('submission.csv', index=False)

print("提交檔案 'submission.csv' 已生成！")
