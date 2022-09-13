from torch.utils.data import Dataset
import os
import numpy as np


class DatasetEnergy(Dataset):
    def __init__(self, data_dir, data_name, source_len=168, is_train=True) -> None:
        """
        data_name  在 {'electricity', 'wind', 'solar', 'price', 'load'} 选
        """
        super().__init__()
        self.is_train = is_train

        data_path = os.path.join(data_dir, data_name)

        if is_train:
            file_name = f'train_data_{data_name}.npy'
            self.data = np.load(os.path.join(data_path, file_name))

        else:
            file_name_data = f'test_data_{data_name}.npy'
            file_name_v = f'test_v_{data_name}.npy'

            self.data = np.load(os.path.join(data_path, file_name_data))
            self.v = np.load(os.path.join(data_path, file_name_v))

        self.source_data = self.data[:, :-1, :source_len]  # 需要预测的值和时变协变量
        target_data = self.data[:, 0, source_len:]
        self.target_data = np.expand_dims(target_data, 1)  # 取单行会造成维度缩减，这里要把维度增加回来
        self.time_cov_future = self.data[:, 1:5, source_len:]
        self.static_cov = self.data[:, 5, 0].reshape(self.data.shape[0], 1, 1)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        if self.is_train:
            x = (self.source_data[index, :, :],
                 self.time_cov_future[index, :, :],
                 self.static_cov[index, :, :],
                 self.target_data[index, :, :])

        else:
            x = (self.source_data[index, :, :],
                 self.time_cov_future[index, :, :],
                 self.static_cov[index, :, :],
                 self.target_data[index, :, :],
                 self.v[index, :])
        return x
