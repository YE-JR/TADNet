from utils.utils import *
from utils.metrics import *

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

import os
import logging
import time

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger('TADnet.train')


def train(model: nn.Module,
          optimizer: optim,
          loss_fn,
          train_loader,
          device,
          epoch: int,
          writer: SummaryWriter):
    model.train()
    loss_epoch = np.zeros(len(train_loader))

    for i, x in enumerate(tqdm(train_loader)):
        # dataloader出来的数据格式均为(batch,feature, seq_len)
        source_data, time_cov_future, static_cov, labels = x

        # 设置设备
        source_data = source_data.to(device)
        time_cov_future = time_cov_future.to(device)
        static_cov = static_cov.int().to(device)
        labels = labels.to(device)

        # 前向
        y_hat = model(source_data=source_data,
                      time_cov_future=time_cov_future,
                      static_cov=static_cov)

        # 反传 + 参数更新
        optimizer.zero_grad()
        loss = loss_fn(y_hat, labels)
        loss.backward()
        optimizer.step()

        # 记录训练过程数据
        loss_epoch[i] = loss.item()
        writer.add_scalar('train loss',
                          loss_epoch[i],
                          epoch * len(train_loader) + i)

    return loss_epoch


def test(model: nn.Module,
         loss_fn,
         test_loader,
         epoch: int,
         device,
         writer: SummaryWriter):
    loss_epoch = np.zeros(len(test_loader))
    model.eval()
    with torch.no_grad():
        # 初始化一些需要记录的性能指标(ND和RMSE)
        raw_metrics = init_metrics()

        for i, x in enumerate(tqdm(test_loader)):
            source_data, time_cov_future, static_cov, labels, v = x
            batch_size = source_data.shape[0]

            # 输入模型的数据
            # 特别说明：
            # v 是用于放缩数据的
            # 放缩因子 v 仅由每次输入模型的历史观测值计算得到
            # 不会造成信息泄露！
            source_data = source_data.to(device)
            time_cov_future = time_cov_future.to(device)
            static_cov = static_cov.int().to(device)
            labels = labels.to(device)
            v = v.to(device)

            y_hat = model(source_data=source_data,
                          time_cov_future=time_cov_future,
                          static_cov=static_cov)

            loss = loss_fn(y_hat, labels)
            loss_epoch[i] = loss.item()
            writer.add_scalar('test loss',
                              loss_epoch[i],
                              epoch * len(test_loader) + i)

            # 用于计算性能指标，放缩回原始尺度之后再进行性能计算
            y_hat_input = (y_hat.permute(1, 2, 0) * v[:, 0] + v[:, 1]).permute(2, 0, 1)  # (batch, q_num, seq_len)
            y_q10 = y_hat_input[:, 0, :]
            y_q50 = y_hat_input[:, y_hat_input.shape[1] // 2, :]
            y_q90 = y_hat_input[:, -1, :]
            labels_input = (labels.squeeze(1).permute(1, 0) * v[:, 0] + v[:, 1]).permute(1, 0)
            ###########################
            raw_metrics = update_metrics(raw_metrics, y_q10, y_q50, y_q90, labels_input)

        summary_metric = final_metrics(raw_metrics)

        writer.add_scalar('summary ND',
                          summary_metric['ND'],
                          epoch)

        metrics_string = '; '.join('{}: {:05.3f}'.format(k, v) for k, v in summary_metric.items())
        logger.info('Full test metrics: ' + metrics_string)

    return loss_epoch, summary_metric


def train_and_test(model: nn.Module,
                   train_loader: DataLoader,
                   test_loader: DataLoader,
                   optimizer: optim,
                   scheduler: optim,
                   loss_fn,
                   num_epochs,
                   device,
                   exp_result_dir,
                   params_dict):
    # 生成训练记录文件夹
    time_now = time.strftime("%Y%m%d-%H%M", time.localtime())
    result_dir = os.path.join(exp_result_dir, time_now)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    #  设置记录日志
    logger_name = 'TADNet'
    logger = logging.getLogger(logger_name)
    set_logger(logger, os.path.join(result_dir, logger_name + '.log'))

    model_params = ''
    for k, v in params_dict.items():
        model_params += f'{k}:{v} \n '

    logger.info(f'model_params: \n {model_params}')
    logger.info('Begin train and test')

    # 使用TensorBoard
    writer = SummaryWriter(result_dir)

    # 训练
    best_test_ND = float('inf')
    # best_test_lose = float('inf')
    best_epoch = 0
    best_metric = {}

    for epoch in range(num_epochs):

        logger.info(f'epoch:{epoch} training...')
        loss_train = train(model, optimizer, loss_fn, train_loader, device, epoch, writer)
        logger.info(f'epoch:{epoch} loss_train={loss_train.mean()}')
        scheduler.step(loss_train.mean())

        logger.info(f'epoch:{epoch} test...')
        loss_test, metrics = test(model, loss_fn, test_loader, epoch, device, writer)
        logger.info(f'epoch:{epoch} loss_test={loss_test.mean()}')

        if metrics['ND'] <= best_test_ND:

            best_epoch = epoch
            best_metric = metrics
            best_test_ND = metrics['ND']
            model_path = os.path.join(result_dir, 'best_model')
            torch.save(model.state_dict(), model_path)
            metrics_string = ';   '.join(' \n {}: {:05.3f}'.format(k, v) for k, v in metrics.items())
            logger.info('Current best metrics: {}, produced in epoch: {}'.format(metrics_string, best_epoch))

        logger.info('Current Best loss is:  {:05.3f}, produced in epoch: {} '.format(best_test_ND, best_epoch))

    logger.info('\n ********** End ************')
    metrics_string = ';  '.join('\n {}: {:05.3f}'.format(k, v) for k, v in best_metric.items())
    logger.info('\n The best metrics: {}, produced in epoch: {}'.format(metrics_string, best_epoch))
