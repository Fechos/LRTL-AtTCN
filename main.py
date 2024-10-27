import os
import sys
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch.backends.cudnn as cudnn
import torch.optim as optim
from LR_AtTCN import At_TCN, LoRA
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import *

def get_shape(lst):
    if isinstance(lst, list):
        return [len(lst)] + get_shape(lst[0])
    else:
        return []


def pre_train(filename, num_epochs):
    best_val_loss = float('inf')
    train_loader, test_loader, valid_loader, scaler = create_dataloader(
        filename, pre_len, window_size, target, feature_cols, device, batch_size=batch_size)
    model = At_TCN(num_inputs=len(feature_cols), outputs=1, pre_len=pre_len, num_channels=[16, 32, 64],
                   dropout=0.2).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 训练
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in valid_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(valid_loader.dataset)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'pre_best_model.pth')
    # 加载最优模型权重
    model.load_state_dict(torch.load('pre_best_model.pth'))
    model.to(device)  # 确保模型在 GPU 上

    # 测试阶段
    model.eval()
    test_loss = 0.0
    predictions = []  # 存储预测值
    targets_list = []  # 存储真实值

    with torch.no_grad():
        for batch in test_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            # 计算损失
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)

            # 将预测值和真实值存入列表，保持时间序列顺序
            predictions.append(outputs.cpu().numpy())  # 转为 numpy 数组并存储
            targets_list.append(targets.cpu().numpy())  # 同样处理真实值

    # 将 predictions 和 targets_list 转换为单个 numpy 数组
    predictions = np.concatenate(predictions, axis=0).squeeze()   # 拼接所有 batch 的预测结果
    targets_list = np.concatenate(targets_list, axis=0).squeeze()   # 拼接所有 batch 的真实值

    # 打印预测和真实值的形状
    print(f"Predictions shape: {predictions.shape}")
    print(f"Targets shape: {targets_list.shape}")

    # 计算性能指标
    mae = mean_absolute_error(predictions, targets_list)
    mse = mean_squared_error(predictions, targets_list)
    r2 = r2_score(predictions, targets_list)
    print(f"Test MAE: {mae:.5f}")
    print(f"     MSE: {mse:.5f}")
    print(f"      R2: {r2:.5f}")


def fine_tune(filename, num_epochs):
    # 加载新数据集的 DataLoader
    train_loader, test_loader, valid_loader, scaler = create_dataloader(
        filename, pre_len, window_size, target, feature_cols, device, batch_size=batch_size)

    # 加载预训练模型
    model = At_TCN(num_inputs=len(feature_cols), outputs=1, pre_len=pre_len, num_channels=[16, 32, 64],
                   dropout=0.2).to(device)
    model.load_state_dict(torch.load('models/pre_best_model.pth'))

    # 设置模型为训练模式，确保权重可学习
    model.train()

    # 定义 LoRA 适配器
    lora_adapter = LoRA(in_features=1, out_features=1, rank=4).to(device)

    # 优化器，可以选择只优化 LoRA 适配器或者同时优化模型
    optimizer = torch.optim.Adam(lora_adapter.parameters(), lr=0.05)
    criterion = nn.MSELoss()

    # 微调阶段
    for epoch in range(num_epochs):
        model.train()
        lora_adapter.train()
        train_loss = 0.0

        for batch in train_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)
            outputs = lora_adapter(outputs)  # 应用 LoRA 适配器

            # 计算损失
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}")
    torch.save(lora_adapter.state_dict(), 'models/finetuned.pth')

    # 加载微调后的模型进行测试
    model.eval()
    lora_adapter.eval()
    test_loss = 0.0
    predictions = []
    targets_list = []  # 存储真实值

    with torch.no_grad():
        for batch in test_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            outputs = lora_adapter(outputs)  # 应用 LoRA 适配器
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)
            # 将预测值和真实值存入列表，保持时间序列顺序
            predictions.append(outputs.cpu().numpy())  # 转为 numpy 数组并存储
            targets_list.append(targets.cpu().numpy())  # 同样处理真实值

    test_loss /= len(test_loader.dataset)
    # 将 predictions 和 targets_list 转换为单个 numpy 数组
    predictions = np.concatenate(predictions, axis=0).squeeze()  # 拼接所有 batch 的预测结果
    targets_list = np.concatenate(targets_list, axis=0).squeeze()  # 拼接所有 batch 的真实值

    # 打印预测和真实值的形状
    print(f"Predictions shape: {predictions.shape}")
    print(f"Targets shape: {targets_list.shape}")

    # 计算性能指标
    mae = mean_absolute_error(predictions, targets_list)
    mse = mean_squared_error(predictions, targets_list)
    r2 = r2_score(predictions, targets_list)
    print(f"Test MAE: {mae:.5f}")
    print(f"     MSE: {mse:.5f}")
    print(f"      R2: {r2:.5f}")


if __name__ == "__main__":
    model_root = 'models'
    cuda = True
    cudnn.benchmark = True
    batch_size = 64
    pre_len = 1
    window_size = 24
    # 'temperature', 'dew_point', 'humidity', 'air_pressure', 'wind_speed', 'hour_of_day', 'is_holiday',
    feature_cols = ['temperature', 'dew_point', 'humidity', 'air_pressure', 'wind_speed',  'OT']
    target = "OT"
    device = torch.device('cuda') if torch.cuda.is_available() \
        else torch.device('cpu')

    # 加载数据，生成dataloader
    data_path = "data/final_data/"

    source_data = f"{data_path}train_W_No0.csv"
    pre_train(source_data, 50)

    target_data = f"{data_path}/min/1_month.csv"
    fine_tune(target_data, 20)



