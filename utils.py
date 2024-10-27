import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import Dataset, DataLoader


class StandardScaler:
    def __init__(self):
        self.mean = 0.
        self.std = 1.

    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        if data.shape[-1] != mean.shape[-1]:
            mean = mean[-1:]
            std = std[-1:]
        return (data * std) + mean


class TimeSeriesDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence, label = self.sequences[index]
        return torch.Tensor(sequence), torch.Tensor(label)


class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        """
        初始化 EarlyStopping.
        :param patience: 在停止之前允许的验证损失没有改善的epoch数.
        :param delta: 改善验证损失的最小变化.
        """
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        将模型保存为最佳模型。
        """
        self.best_model = model.state_dict()


def calculate_mae(y_true, y_pred):
    # 平均绝对误差
    mae = np.mean(np.abs(y_true - y_pred))
    return mae


def calculate_mape(y_true, y_pred):
    # 平均绝对百分比误差
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mape


def calculate_cv(y_true, y_pred):
    # 计算预测误差
    residuals = y_true - y_pred

    # 计算预测误差的标准差
    std_residuals = np.std(residuals)

    # 计算真实值的均值
    mean_true = np.mean(y_true)

    # 计算变异系数
    cv = std_residuals / mean_true

    return cv


def calculate_rmse(y_true, y_pred):
    # 均方根误差
    # 选择每个样本的最后一个时间步的预测值和真实值
    y_true_last = y_true
    y_pred_last = y_pred

    # 计算均方根误差
    rmse = np.sqrt(mean_squared_error(y_true_last, y_pred_last))
    return rmse


def calculate_r2(y_true, y_pred):
    # R^2 拟合优度
    r2 = r2_score(y_true, y_pred)
    return r2


def log_mae(y_true, y_pred):
    absolute_errors = np.abs(y_true - y_pred)
    log_absolute_errors = np.log(absolute_errors + 1e-8)  # 加上一个小值避免对数为无穷大
    return np.mean(log_absolute_errors)


def normalized_mse(y_true, y_pred):
    """
    计算归一化均方误差（NMSE）

    参数:
    y_true -- 真实值数组
    y_pred -- 预测值数组

    返回值:
    nmse -- 归一化均方误差
    """
    # 计算均方误差（MSE）
    mse = mean_squared_error(y_true, y_pred)

    # 计算真实值的方差
    variance = np.var(y_true)

    # 计算归一化的均方误差（NMSE）
    nmse = 1 - mse / variance

    return nmse

def create_inout_sequences(input_data, tw, pre_len, feature="MS"):
    # 创建时间序列数据专用的数据分割器
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i:i + tw]
        if (i + tw + pre_len) > len(input_data):
            break
        if feature == 'MS':
            train_label = input_data[:, -1:][i + tw:i + tw + pre_len]
        else:
            train_label = input_data[i + tw:i + tw + pre_len]
        inout_seq.append((train_seq, train_label))
    return inout_seq

def create_dataloader(data_path, pre_len, window_size, target, feature_cols, device, batch_size, debug=False):
    df = pd.read_csv(data_path)  # 填你自己的数据地址,自动选取你最后一列数据为特征列 # 添加你想要预测的特征列
    pre_len = pre_len  # 预测未来数据的长度
    train_window = window_size  # 观测窗口

    # 将特征列移到末尾
    target_data = df[[target]]
    df = df.drop(target, axis=1)
    df = pd.concat((df, target_data), axis=1)

    cols_data = feature_cols
    df_data = df[cols_data]

    # 这里加一些数据的预处理, 最后需要的格式是pd.series
    true_data = df_data.values

    # 定义标准化优化器
    # 定义标准化优化器
    scaler = StandardScaler()
    scaler.fit(true_data)

    train_data = true_data[:int(0.7 * len(true_data))]
    valid_data = true_data[int(0.7 * len(true_data)):int(0.85 * len(true_data))]
    test_data = true_data[int(0.85 * len(true_data)):]

    # 进行标准化处理
    train_data_normalized = scaler.transform(train_data)
    test_data_normalized = scaler.transform(test_data)
    valid_data_normalized = scaler.transform(valid_data)

    # 转化为深度学习模型需要的类型Tensor
    train_data_normalized = torch.FloatTensor(train_data_normalized).to(device)
    test_data_normalized = torch.FloatTensor(test_data_normalized).to(device)
    valid_data_normalized = torch.FloatTensor(valid_data_normalized).to(device)

    # 定义训练器的的输入
    train_inout_seq = create_inout_sequences(train_data_normalized, train_window, pre_len)
    test_inout_seq = create_inout_sequences(test_data_normalized, train_window, pre_len)
    valid_inout_seq = create_inout_sequences(valid_data_normalized, train_window, pre_len)

    # 创建数据集
    train_dataset = TimeSeriesDataset(train_inout_seq)
    test_dataset = TimeSeriesDataset(test_inout_seq)
    valid_dataset = TimeSeriesDataset(valid_inout_seq)

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    if debug:
        print("训练集尺寸:", len(train_data), "测试集尺寸:", len(test_data), "验证集尺寸:", len(valid_data))
        print("通过滑动窗口共有训练集数据：", len(train_inout_seq), "转化为批次数据:", len(train_loader))
        print("通过滑动窗口共有测试集数据：", len(test_inout_seq), "转化为批次数据:", len(test_loader))
        print("通过滑动窗口共有验证集数据：", len(valid_inout_seq), "转化为批次数据:", len(valid_loader))
    return train_loader, test_loader, valid_loader, scaler

class CombinedLoader(DataLoader):
    def __init__(self, loaders):
        # 确保传入的是DataLoader对象列表
        if not all(isinstance(loader, DataLoader) for loader in loaders):
            raise ValueError("All elements in loaders must be instances of DataLoader")
        self.loaders = loaders
        self.loader_iters = [iter(loader) for loader in loaders]
        super().__init__(dataset=None)  # 父类初始化，可以根据需要调整

    def __iter__(self):
        self.loader_iters = [iter(loader) for loader in self.loaders]
        return self

    def __next__(self):
        batches = []
        for loader_iter in self.loader_iters:
            try:
                batch = next(loader_iter)
                batches.append(batch)
            except StopIteration:
                self.loader_iters.remove(loader_iter)
                if not self.loader_iters:
                    raise StopIteration
        return batches

    def __len__(self):
        return min(len(loader) for loader in self.loaders)