from model.submodel import *
import torch
import torch.nn as nn


class BasicsBlock(nn.Module):
    def __init__(self, cnn_num_inputs: int, num_channels: list, dropout,
                 static_cov_dim: list, hidden_size, quantiles_num,
                 num_heads, source_len, target_len, is_Regular=True, fourier_P=None) -> None:
        """
        fourier_P：当设置为傅里叶基底时, P为模型考虑的最大周期(必须取偶数)
        """
        super().__init__()

        self.fourier_P = fourier_P
        self.quantiles_num = quantiles_num

        self.cnn_block = TemporalConvNet(num_inputs=cnn_num_inputs, num_channels=num_channels, dropout=dropout)
        self.process_cov = ProcessStatic(static_cov_dim, hidden_size)
        self.grn = GRN(num_channels[-1], hidden_size)

        self.attention = AttentionNet(model_dim=hidden_size,
                                      num_heads=num_heads,
                                      source_len=source_len)

        if is_Regular:
            # Regular_Block
            if fourier_P is not None:
                # 采用傅里叶基
                self.last_layer = nn.Sequential(nn.Linear(hidden_size, fourier_P),
                                                SeasonalBasis(target_len, fourier_P))
            else:
                # 采用可学习基
                self.last_layer = nn.Linear(hidden_size, 1)
        else:
            self.last_layer = nn.Linear(hidden_size, quantiles_num)

    def forward(self, source_data, static_embedding, feature_future=None):
        """
        输出的长度为需要预测的长度
        """
        cnn_output = self.cnn_block(source_data).permute(2, 0, 1)  # (seq_len, batch, feature)
        feature_past = self.grn(cnn_output, static_embedding)

        if feature_future is not None:

            # 规律项
            attn_input = torch.concat((feature_past, feature_future), dim=0)
            attn_output = self.attention(attn_input)
            if self.fourier_P is not None:
                output = self.last_layer(attn_output)
            else:
                output = self.last_layer(attn_output).permute(1, 2, 0)
        else:
            # 残余项
            # 噪声项的cnn_output已经包含了未来的信息（经由上一步预测所得）
            attn_input = feature_past
            attn_output = self.attention(attn_input)
            temp = self.last_layer(attn_output).permute(1, 2, 0)  # (batch, feature, seq_len)

            # 处理残余项输出的波动正负（波动下界为强制转负，波动上界转正）
            output_list = list()
            for i in range(self.quantiles_num):
                if i < self.quantiles_num // 2:
                    output_list.append(torch.unsqueeze(- torch.abs(temp[:, i, :]), dim=1))
                elif i == self.quantiles_num // 2:
                    output_list.append(torch.unsqueeze(temp[:, i, :], dim=1))
                else:
                    output_list.append(torch.unsqueeze(torch.abs(temp[:, i, :]), dim=1))
            output = torch.cat(output_list, dim=1)

        return output


class TADNet(nn.Module):
    def __init__(self, cnn_num_inputs, num_channels, dropout, static_cov_dim,
                 hidden_size, num_time_cov, num_heads, source_len,
                 target_len, quantiles_num, fourier_P=None) -> None:
        super().__init__()
        self.regular = None
        self.remainder = None

        self.regular_block = BasicsBlock(cnn_num_inputs=cnn_num_inputs,
                                         num_channels=num_channels,
                                         dropout=dropout,
                                         static_cov_dim=static_cov_dim,
                                         hidden_size=hidden_size,
                                         quantiles_num=quantiles_num,
                                         num_heads=num_heads,
                                         source_len=source_len,
                                         target_len=target_len,
                                         fourier_P=fourier_P)

        self.residual_block = BasicsBlock(cnn_num_inputs=cnn_num_inputs,
                                          num_channels=num_channels,
                                          dropout=dropout,
                                          static_cov_dim=static_cov_dim,
                                          hidden_size=hidden_size,
                                          quantiles_num=quantiles_num,
                                          num_heads=num_heads,
                                          source_len=source_len,
                                          target_len=target_len,
                                          is_Regular=False
                                          )

        self.process_static = ProcessStatic(static_cov_dim, hidden_size)
        self.fc_cov_future = nn.Linear(num_time_cov, hidden_size)

    def forward(self, source_data, time_cov_future, static_cov=None):
        """
        输出变量的维度(batch, feature, seq_len)
        """
        # 协变量处理
        static_embedding = self.process_static(static_cov)  # (1, batch, embedding_size)
        feature_future = self.fc_cov_future(time_cov_future.permute(2, 0, 1))
        # 规律项
        regular = self.regular_block(source_data, static_embedding, feature_future)  # (batch, feature, seq_len)
        # 残余项
        residual_input_future = torch.concat((regular, time_cov_future), dim=1)
        residual_input = torch.concat((source_data, residual_input_future), dim=-1)
        residual = self.residual_block(residual_input, static_embedding)

        # 输出
        output = regular + residual
        self.regular = regular
        self.remainder = residual

        return output

    def split(self):
        return self.regular, self.remainder


# 损失

class QuantileLoss(nn.Module):
    # From: https://medium.com/the-artificial-impostor/quantile-regression-part-2-6fdbc26b2629

    def __init__(self, quantiles=[0.1, 0.5, 0.9]):
        """
        quantiles: list, 分位数损失
        """
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        """
        (batch, quantiles_num, seq_len)
        """
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        error = target - preds
        losses = []
        for i, q in enumerate(self.quantiles):
            losses.append(torch.max((q - 1) * error[:, i, :], q * error[:, i, :]))
        loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
        return loss
