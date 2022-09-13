import os
import numpy as np
from scipy import stats
import pandas as pd
import logging
from tqdm import trange

logger = logging.getLogger('TADNet.preprocess_data')


def prepocess_data(data, covariates, data_start, train=True):
    """
    对数据进行预处理并保存（方便数据复用，使得 Dataset 可以写得更加简单）
    time_len ： 数据集的时间总长度（去掉前面的0之后的长度）
    total_time ： 数据集没去0的长度
    seq_len : 输入模型的序列总长度，历史观测数据与需要预测的序列的长度之和
    target_len ： 需要预测的序列的长度
    num_covariates ： 需要使用的协变量的总个数（包括了静态协变量和时变协变量）
    """
    time_len = data.shape[0]  # 输入数据集的总的时间长度
    input_size = window_size - stride_size

    # np.full：生成全是指定数值的指定形状的array
    # windows_per_series：统计各个用户的用于训练的样本数量
    # 因为窗口是用七天的数据去预测一天的；而且窗口是一天一天往前滚的
    # 所以：窗口数量为：（时间总长 - 用于预测的历史数据长度）// （需要预测的数据长度）
    windows_per_series = np.full((series_num), (time_len - input_size) // stride_size)

    if train: windows_per_series -= (data_start + stride_size - 1) // stride_size

    total_windows = np.sum(windows_per_series)

    # 初始化用于存储 输入x_input，用于放缩的均值v, 真值label的矩阵
    # x_input = np.zeros((total_windows, window_size, 1 + num_covariates + 1), dtype='float32')
    # x_input = (batch, feature, seqlen)
    x_input = np.zeros((total_windows, 1 + num_covariates + 1, window_size), dtype='float32')
    v_input = np.zeros((total_windows, 2), dtype='float32')

    # cov = 3: ground truth + age + day_of_week + hour_of_day + num_series
    # cov = 4: ground truth + age + day_of_week + hour_of_day + month_of_year + num_series

    # num_series = 59
    # cov_age:时间步归一化之后的列
    # 协变量：time_step; week; hour; month;
    count = 0
    # count:记录有多少个sample
    if not train:
        # 在之前存储协变量时，是整个数据集一起存储的
        # 因此在测试集时。需要取后面时间的协变量
        covariates = covariates[-time_len:]

    for series in trange(series_num):
        # num_series可以视为列数，列数也就是用户的数量
        # cov_age:time_step归一化
        cov_age = stats.zscore(np.arange(total_time - data_start[series]))
        if train:
            covariates[data_start[series]:time_len, 0] = cov_age[:time_len - data_start[series]]
        else:
            covariates[:, 0] = cov_age[-time_len:]

        for i in range(windows_per_series[series]):
            # 对单个用户的窗口进行循环
            if train:
                window_start = stride_size * i + data_start[series]
            else:
                window_start = stride_size * i
            window_end = window_start + window_size

            x_input[count, 0, :] = data[window_start:window_end, series]  # 需要预测的值
            x_input[count, 1:1 + num_covariates, :] = np.transpose(covariates[window_start:window_end, :])  # 时变协变量
            x_input[count, -1, :] = int(series)  # 用户id（静态协变量）

            # 检测时序数据中有多少非零值
            # v用于存储（mean和std）
            # 对变量（用电量）进行放缩的方式就是直接除以均值
            nonzero_sum = (x_input[count, 0, 1:input_size] != 0).sum()
            if nonzero_sum == 0:
                v_input[count, 0] = 0
            else:
                v_input[count, 0] = np.true_divide(x_input[count, 0, 1:input_size].sum(), nonzero_sum) + 1
                x_input[count, 0, :] = x_input[count, 0, :] / v_input[count, 0]
            count += 1

    prefix = os.path.join(save_path, 'train_' if train else 'test_')
    np.save(prefix + 'data_' + data_name, x_input)
    np.save(prefix + 'v_' + data_name, v_input)


def gen_covariates(times, num_covariates):
    covariates = np.zeros((times.shape[0], num_covariates))
    for i, input_time in enumerate(times):
        covariates[i, 1] = input_time.weekday()
        covariates[i, 2] = input_time.hour
        covariates[i, 3] = input_time.month
    for i in range(1, num_covariates):
        covariates[:, i] = stats.zscore(covariates[:, i])
    return covariates[:, :num_covariates]


pass

if __name__ == '__main__':
    data_name = 'wind'  # {electricity; load; price; solar; wind} 五选一
    num_covariates = 4
    window_size = 192
    stride_size = 24

    if data_name == 'electricity':
        train_start = '2012-01-01 00:00:00'
        train_end = '2014-06-07 23:00:00'
        test_start = '2014-06-01 00:00:00'  # need additional 7 days as given info
        test_end = '2014-12-31 23:00:00'

    else:
        train_start = '2015-01-01 00:00:00'
        train_end = '2017-6-30 23:00:00'
        test_start = '2017-6-24 00:00:00'  # need additional 7 days as given info
        test_end = '2017-11-30 23:00:00'

    csv_path = os.path.join("data\\" + data_name, data_name + ".csv.gz")
    save_path = "data\\" + data_name

    if not os.path.exists(csv_path):
        logger.error('数据集不存在！')

    data_frame = pd.read_csv(csv_path, sep=",", index_col=0, parse_dates=True, decimal='.')

    data_frame.fillna(0, inplace=True)
    covariates = gen_covariates(data_frame[train_start:test_end].index, num_covariates)
    train_data = data_frame[train_start:train_end].values

    test_data = data_frame[test_start:test_end].values
    data_start = (train_data != 0).argmax(axis=0)  # find first nonzero value in each time series

    total_time = data_frame.shape[0]
    series_num = data_frame.shape[1]

    prepocess_data(train_data, covariates, data_start)
    prepocess_data(test_data, covariates, data_start, train=False)
