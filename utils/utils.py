import logging
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import json

logger = logging.getLogger('TADNet.Utils')


################# 参数设置 ###################
class Params:
    """Class that loads hyperparameters from a json file.
    Example:
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4, ensure_ascii=False)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by params.dict['learning_rate']"""
        return self.__dict__


################## 画图 ########################
def plot_windows(plot_dir,
                 y_q10,
                 y_q50,
                 y_q90,
                 labels,
                 window_size,
                 predict_start,
                 plot_num,
                 plot_metrics):
    window = np.arange(window_size)

    f = plt.figure(figsize=(8, 42), constrained_layout=True)
    nrows = 21
    ncols = 1
    ax = f.subplots(nrows, ncols)

    for k in range(nrows):
        if k == 10:
            ax[k].plot(window, window, color='g')
            ax[k].plot(window, window[::-1], color='g')
            ax[k].set_title('This separates top 10 and bottom 90', fontsize=10)
            continue
        m = k if k < 10 else k - 1

        ax[k].plot(window[predict_start:], y_q50[m], color='b', label='predict')
        ax[k].fill_between(window[predict_start:], y_q10[m], y_q90[m], color='blue', alpha=0.2)
        ax[k].plot(window, labels[m, :], color='r', label='true')
        ax[k].axvline(predict_start, color='g', linestyle='dashed')
        ax[k].grid()
        ax[k].legend(prop={'family': 'Times New Roman', 'size': 10})

        # metrics = utils.final_metrics_({_k: [_i[k] for _i in _v] for _k, _v in plot_metrics.items()})
        plot_metrics_str = f'ND: {plot_metrics["ND"][m]: .3f} ' \
                           f'RMSE: {plot_metrics["RMSE"][m]: .3f}'

        ax[k].set_title(plot_metrics_str, fontsize=10)

    f.savefig(os.path.join(plot_dir, str(plot_num) + '.png'))
    plt.close()


################################################


############# 设置训练记录logger ################
def set_logger(_logger, log_path):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    logging.info('Starting training...')
    Args:
        log_path: (string) where to log
    """

    # _logger = logging.getLogger(logger_name)
    _logger.setLevel(logging.INFO)

    fmt = logging.Formatter('[%(asctime)s] %(name)s: %(message)s', '%D  %H:%M:%S')

    class TqdmHandler(logging.StreamHandler):
        def __init__(self, formatter):
            logging.StreamHandler.__init__(self)
            self.setFormatter(formatter)

        def emit(self, record):
            msg = self.format(record)
            tqdm.write(msg)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(fmt)
    _logger.addHandler(file_handler)
    _logger.addHandler(TqdmHandler(fmt))
