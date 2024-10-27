import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

# LoRA模块
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super(LoRA, self).__init__()
        self.rank = rank
        self.in_features = in_features
        self.out_features = out_features
        self.A = nn.Parameter(torch.randn(out_features, rank))
        self.B = nn.Parameter(torch.randn(rank, in_features))

    def forward(self, x):
        return x @ self.A @ self.B

# Chomp1d模块
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

# TemporalBlock模块
class At_TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, n_heads=4):
        super(At_TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.attention = nn.MultiheadAttention(embed_dim=n_outputs, num_heads=n_heads)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)

        # Applying self-attention
        out = out.permute(2, 0, 1)  # Change shape to (seq_len, batch_size, n_outputs)
        out, _ = self.attention(out, out, out)
        out = out.permute(1, 2, 0)  # Change back to (batch_size, n_outputs, seq_len)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

# TemporalConvNet模型
class At_TCN(nn.Module):
    def __init__(self, num_inputs, outputs, pre_len, num_channels, kernel_size=2, dropout=0.2, lora_rank=4):
        super(At_TCN, self).__init__()
        layers = []
        self.pre_len = pre_len
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.lora_rank = lora_rank

        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [At_TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                        padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], outputs)
        self.lora = LoRA(outputs, outputs, rank=lora_rank)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.network(x)
        x = x.permute(0, 2, 1)
        x = self.linear(x)
        x = self.lora(x)
        return x[:, -self.pre_len:, :]

    def count_parameters(self):
        total_params = 0

        # Count parameters in temporal blocks
        for i in range(len(self.num_channels)):
            in_channels = self.num_channels[i - 1] if i > 0 else self.num_channels[0]  # Input channels for first block
            out_channels = self.num_channels[i]

            # Parameters for conv1
            conv1_params = (in_channels * out_channels * self.kernel_size) + out_channels
            # Parameters for conv2
            conv2_params = (out_channels * out_channels * self.kernel_size) + out_channels

            total_params += conv1_params + conv2_params

        # Count parameters in linear layer
        total_params += (self.num_channels[-1] * self.linear.out_features) + self.linear.out_features

        # Count parameters in LoRA
        total_params += 2 * (self.linear.out_features * self.lora_rank)  # LoRA Parameters

        return total_params

    def count_lora_parameters(self):
        # Count only LoRA parameters
        lora_params = (self.linear.out_features * self.lora_rank) + (self.lora_rank * self.num_channels[-1])  # in_features is num_channels[-1]
        return lora_params


# Example Usage
num_inputs = 8
outputs = 1
pre_len = 1
num_channels = [16, 32, 64]  # Example channel configuration
total_params_list = []
lora_params_list = []

for l in [4, 8, 16, 32, 64]:
    model = At_TCN(num_inputs, outputs, pre_len, num_channels=[16, 32, 64], lora_rank=l)
    total_params = model.count_parameters()
    lora_params = model.count_lora_parameters()

    total_params_list.append(total_params)
    lora_params_list.append(lora_params)

print(total_params_list)
print(lora_params_list)
print(lora_params_list[2]/total_params_list[2])
