import math
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


############################################### 卷积块 ###########################################
class Chomp1d(nn.Module):
    """
    用来做padding的
    """

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        """
        n_inputs:输入变量（batch,channel,seq_len）的channel
        n_outputs:输出变量的通道数，可以理解为输出的特征维度
        """
        super(TemporalBlock, self).__init__()
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
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """
    输出序列的长度与输入序列相等
    n_inputs：输入序列的通道数（维数）
    num_channels：各TCN块的通道数；（最终的TCN网络是由len(num_channels)层TCN块堆叠而成的）
    """

    def __init__(self, num_inputs, num_channels, kernel_size=7, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i  # 层数越深越稀疏
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


################################################# 协变量处理 ####################################################
class GatedLinearUnit(nn.Module):
    def __init__(self, input_size,
                 hidden_size,
                 dropout_rate=None):

        super(GatedLinearUnit, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

        if self.dropout_rate:
            self.dropout = nn.Dropout(p=self.dropout_rate)

        self.W4 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.W5 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        for n, p in self.named_parameters():
            if 'bias' not in n:
                torch.nn.init.xavier_uniform_(p)
            elif 'bias' in n:
                torch.nn.init.zeros_(p)

    def forward(self, x):
        """
        输入的维度：(time_step, batch, feature)
        """
        if self.dropout_rate:
            x = self.dropout(x)
        output = self.sigmoid(self.W4(x)) * self.W5(x)
        return output


class ProcessStatic(nn.Module):
    """
    用来预处理静态协变量，得到的输出用于辅助时间协变量的选择

    static_cov_num：一个列表，static_cov_num[i]:表示第i个静态协变量的分类数目
    static_cov_num: 静态协变量：不随时间的改变而改变的变量
                    如不同客户的id,以唯一对应的数字的形式送进来即可
                    维数：(batch, static_cov_num, 1)
    """

    def __init__(self,
                 static_cov_num,
                 embedding_size,
                 dropout_rate=None) -> None:
        super().__init__()

        self.static_cov_num = static_cov_num
        self.embedding_list = nn.ModuleList()  # 用来对静态协变量做embedding
        for i in range(len(static_cov_num)):
            self.embedding_list.append(nn.Embedding(static_cov_num[i], embedding_size))

        self.glu = GatedLinearUnit(len(static_cov_num) * embedding_size,
                                   embedding_size, dropout_rate)

    def forward(self, x):
        """
        x : (batch, len(static_cov_num), 1)
        output:(batch,1,embedding_size)
        静态协变量的seq_len = 1
        """
        static_embedding = []
        for i in range(len(self.static_cov_num)):
            static_embedding.append(self.embedding_list[i](x[:, i, :]))
        static_embedding_cat = torch.concat(static_embedding, -1)
        static_embedding = self.glu(static_embedding_cat)
        return static_embedding.permute(1, 0, 2)  # (seq_len, batch, embedding_size)


class GRN(nn.Module):
    def __init__(self, input_size, hidden_size) -> None:
        """
        符号参考tft这篇论文
        input_size: 输入x的特征维度
        static_cov的特征维度embedding_size = hidden_size
        """
        super().__init__()
        self.dowmsample = None

        if input_size != hidden_size:
            self.dowmsample = nn.Linear(input_size, hidden_size)

        self.hidden_size = hidden_size
        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.W2 = nn.Linear(input_size, hidden_size)  # 用以处理x
        self.W3 = nn.Linear(hidden_size, hidden_size)  # 用以处理static_embedding
        self.elu = nn.ELU()
        self.glu = GatedLinearUnit(hidden_size, hidden_size)
        self.layernorm = nn.LayerNorm(hidden_size)

    def forward(self, x, static_embedding):
        """
        x = (seq_len, batch, feature_size)
        """
        if self.dowmsample is not None:
            res = self.dowmsample(x)
        else:
            res = x
        eta2 = self.elu(self.W2(x) + self.W3(static_embedding))
        eta1 = self.W1(eta2)
        output = self.layernorm(res + self.glu(eta1))

        return output


######################################### 注意力网络 ############################################
class PositionalEmbedding(nn.Module):
    """
    位置信息嵌入
    attention的输入维度(seq_len, batch, model_dim)
    位置信息要与attention的维度匹配
    """

    def __init__(self, model_dim, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, model_dim).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, model_dim, 2).float() * -(math.log(10000.0) / model_dim)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)  # (seq_len, model_dim)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).permute(1, 0, 2)  # (seq_len, batch, model_dim)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:x.shape[0], :, :]


class AttentionNet(nn.Module):
    """
    attention 的本质是获取各个位置之间的关系,不涉及特征维度的变化

    """

    def __init__(self, model_dim, num_heads, source_len) -> None:
        super().__init__()
        # attention的输入维度(seq_len, batch, model_dim)
        # model_dim就是feature的维度
        self.source = source_len
        self.attention = nn.MultiheadAttention(embed_dim=model_dim, num_heads=num_heads, batch_first=False)
        self.pe_embedding = PositionalEmbedding(model_dim=model_dim)
        self.fc_key = nn.Linear(model_dim, model_dim)
        self.fc_value = nn.Linear(model_dim, model_dim)
        self.layer_norm = nn.LayerNorm(model_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        pos = self.pe_embedding(x)
        x = x + pos
        query = x[self.source:, :, :]
        key = self.fc_key(x[:self.source, :, :])
        value = self.fc_value(x[:self.source, :, :])
        attn_output, _ = self.attention(query, key, value)
        output = self.layer_norm(attn_output)
        return self.relu(output)


######################################### 周期项约束函数 ###############################################

class SeasonalBasis(nn.Module):
    """
    Harmonic functions to model seasonality.
    
    """

    def __init__(self, target_len: int, P=24) -> None:
        """
        P 为周期
        """
        super().__init__()
        N = P // 2
        t = torch.arange(target_len)[None, :]
        i = torch.arange(1, N + 1)[None, :] / P

        theta = i.permute(1, 0) * t
        seasonal_embedding = torch.zeros(N * 2, target_len)
        seasonal_embedding[0::2, :] = torch.sin(theta)
        seasonal_embedding[1::2, :] = torch.cos(theta)
        self.seasonal_embedding = nn.Parameter(seasonal_embedding, requires_grad=False)

    def forward(self, x):
        """
        x = (time_step, batch, feature)
        需要转为(batch, feature, time_step)

        feature = N*2
        """
        x = x.permute(1, 2, 0)
        y = (x * self.seasonal_embedding).sum(1).unsqueeze(1)
        return y
