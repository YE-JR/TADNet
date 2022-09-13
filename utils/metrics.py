import numpy as np
import torch
import math
import logging

logger = logging.getLogger('TADnet.metrics')


# ND：相对误差
def accuracy_ND(mu: torch.Tensor, labels: torch.Tensor, relative=False):
    """
    用于计算整个测试集的ND的中间量记录
    """
    zero_index = (labels != 0)
    if relative:
        diff = torch.mean(torch.abs(mu[zero_index] - labels[zero_index])).item()
        return [diff, 1]
    else:
        diff = torch.sum(torch.abs(mu[zero_index] - labels[zero_index])).item()
        summation = torch.sum(torch.abs(labels[zero_index])).item()

        return [diff, summation]


# RMSE：均方根误差
def accuracy_RMSE(mu: torch.Tensor, labels: torch.Tensor, relative=False):
    zero_index = (labels != 0)
    diff = torch.sum(torch.mul((mu[zero_index] - labels[zero_index]), (mu[zero_index] - labels[zero_index]))).item()
    if relative:
        return [diff, torch.sum(zero_index).item(), torch.sum(zero_index).item()]
    else:
        summation = torch.sum(torch.abs(labels[zero_index])).item()
        if summation == 0:
            logger.error('summation denominator error! ')
        return [diff, summation, torch.sum(zero_index).item()]


# 分位数损失
def accuracy_ROU(q: float, samples: torch.Tensor, labels: torch.Tensor, relative=False):
    """
    input_shape = (batch, time_step)
    """
    zero_index = (labels != 0)
    error = labels[zero_index] - samples[zero_index]

    ql = torch.max((q - 1) * error, q * error)
    numerator = 2 * torch.sum(ql).item()
    denominator = torch.abs(labels[zero_index]).sum().item()

    return [numerator, denominator]


def accuracy_ND_(mu: torch.Tensor, labels: torch.Tensor, relative=False):
    """
    计算一个epoch的ND
    直接得到结果
    """
    mu = mu.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()

    mu[labels == 0] = 0.

    diff = np.sum(np.abs(mu - labels), axis=1)
    if relative:
        summation = np.sum((labels != 0), axis=1)
        mask = (summation == 0)
        summation[mask] = 1
        result = diff / summation
        result[mask] = -1
        return result
    else:
        summation = np.sum(np.abs(labels), axis=1)
        mask = (summation == 0)
        summation[mask] = 1
        result = diff / summation
        result[mask] = -1
        return result


def accuracy_RMSE_(mu: torch.Tensor, labels: torch.Tensor, relative=False):
    # (batch, seq_len)
    mu = mu.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()

    mask = labels == 0
    mu[mask] = 0.

    diff = np.sum((mu - labels) ** 2, axis=1)
    summation = np.sum(np.abs(labels), axis=1)
    mask2 = (summation == 0)
    if relative:
        div = np.sum(~mask, axis=1)
        div[mask2] = 1
        result = np.sqrt(diff / div)
        result[mask2] = -1
        return result
    else:
        summation[mask2] = 1
        result = (np.sqrt(diff) / summation) * np.sqrt(np.sum(~mask, axis=1))
        result[mask2] = -1
        return result


def PICP(low, up, label):
    sample_num = label.shape[0] * label.shape[1]
    inclu = (label >= low) & (label <= up)
    inclu_num = inclu.sum().item()
    return inclu_num, sample_num


def init_metrics():
    metrics = {
        'ND': np.zeros(2),  # numerator, denominator
        'RMSE': np.zeros(3),  # numerator, denominator, time step count
        'rou90': np.zeros(2),  # numerator, denominator
        'rou50': np.zeros(2),  # numerator, denominator
        'PICP': np.zeros(2)
    }
    return metrics


def get_metrics(predict, labels, relative=False):
    metric = dict()
    metric['ND'] = accuracy_ND_(predict, labels, relative=relative)
    metric['RMSE'] = accuracy_RMSE_(predict, labels, relative=relative)
    return metric


def update_metrics(raw_metrics, predict10, predict50, predict90, labels, relative=False):
    raw_metrics['ND'] = raw_metrics['ND'] + accuracy_ND(predict50, labels, relative=relative)
    raw_metrics['RMSE'] = raw_metrics['RMSE'] + accuracy_RMSE(predict50, labels, relative=relative)
    raw_metrics['rou90'] = raw_metrics['rou90'] + accuracy_ROU(0.9, predict90, labels, relative=relative)
    raw_metrics['rou50'] = raw_metrics['rou50'] + accuracy_ROU(0.5, predict50, labels, relative=relative)
    raw_metrics['PICP'] = raw_metrics['PICP'] + PICP(predict10, predict90, labels)

    return raw_metrics


def final_metrics(raw_metrics):
    summary_metric = {}
    summary_metric['ND'] = raw_metrics['ND'][0] / raw_metrics['ND'][1]
    summary_metric['RMSE'] = np.sqrt(raw_metrics['RMSE'][0] / raw_metrics['RMSE'][2]) / (
            raw_metrics['RMSE'][1] / raw_metrics['RMSE'][2])
    summary_metric['rou90'] = raw_metrics['rou90'][0] / raw_metrics['rou90'][1]
    summary_metric['rou50'] = raw_metrics['rou50'][0] / raw_metrics['rou50'][1]

    summary_metric['PICP'] = raw_metrics['PICP'][0] / raw_metrics['PICP'][1]
    return summary_metric
