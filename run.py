import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from data_process.dataset import DatasetEnergy
from utils.utils import Params
from model.TAD_Net import TADNet, QuantileLoss
from train_model import train_and_test

import os
import logging

logger = logging.getLogger('TADNet.train')

'''
运行该程序之前, 请确定已经预处理好数据!
'''

# 选择模型拟合方式、数据集
mode = 'L'  # {'L', 'F'} 二选一，L代表可学习基， F代表傅里叶基
data_name = 'wind'  # {electricity; load; price; solar; wind} 五选一
#######################################################################

switch_static_cov_dim = {'electricity': [321],
                         'load': [59],
                         'price': [31],
                         'wind': [57],
                         'solar': [36]}  # 各数据集中包含的时间序列个数

json_path = 'experiments\\params.json'

# 参数设置
params = Params(json_path)
if mode == 'F':
    params.fourier_P = 720
else:
    params.fourier_P = None
params.static_cov_dim = switch_static_cov_dim[data_name]

params.quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # 根据需求设置分位数输出
params.quantiles_num = len(params.quantiles)  # 根据分位数设置得到分位数输出个数

# 创建文件夹，保存训练信息
exp_dir = 'experiments\\' + data_name
# if not os.path.exists(exp_dir):
#     os.mkdir(exp_dir)
exp_result_dir = os.path.join(exp_dir, mode + '_' + str(params.target_len))
if not os.path.exists(exp_result_dir):
    os.makedirs(exp_result_dir)

# 设置训练设备
cuda_exist = torch.cuda.is_available()
if cuda_exist:
    params.device = torch.device('cuda')
    logger.info('Using Cuda...')

else:
    params.device = torch.device('cpu')
    logger.info('Not using cuda...')

# 实例化模型
model = TADNet(cnn_num_inputs=params.cnn_num_inputs,
               num_channels=params.num_channel,
               dropout=params.dropout,
               static_cov_dim=params.static_cov_dim,
               hidden_size=params.hidden_size,
               num_time_cov=params.num_time_cov,
               num_heads=params.num_heads,
               source_len=params.source_len,
               target_len=params.target_len,
               quantiles_num=params.quantiles_num,
               fourier_P=params.fourier_P).to(params.device)

# 数据加载器
train_dataset = DatasetEnergy(data_dir='data', data_name=data_name, is_train=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=params.batch_size, shuffle=True)

test_dataset = DatasetEnergy(data_dir='data', data_name=data_name, is_train=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=params.batch_size, shuffle=False)

# 优化器、损失函数、学习率调整
optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
loss_fn = QuantileLoss(params.quantiles)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10)

# 训练
train_and_test(model=model,
               train_loader=train_loader,
               test_loader=test_loader,
               optimizer=optimizer,
               scheduler=scheduler,
               loss_fn=loss_fn,
               num_epochs=params.num_epochs,
               device=params.device,
               exp_result_dir=exp_result_dir,
               params_dict=params.__dict__)
