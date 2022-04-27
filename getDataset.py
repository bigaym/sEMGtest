import numpy as np
import os
from sEMGtest.preprocess import *
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
degree_arg = 22.22222222

# 归一化
def normalization_list(datalist):

    M = np.max(datalist)  # 803
    m = np.min(datalist)  # -1545
    return (datalist - m) / (M - m)

# 读取TXT，转换格式保存为Tensor文件
def get_save_tensor():
    windows_length = 50

    workplace = parent = os.path.dirname(os.path.realpath(__file__))
    folders = []
    folders.append(workplace + r"\data\上坡\txt")
    folders.append(workplace + r"\data\下坡\txt")
    folders.append(workplace + r"\data\上楼梯\txt")
    folders.append(workplace + r"\data\下楼梯\txt")
    folders.append(workplace + r"\data\平地\txt")
    x_data = []
    y_data = []

    # 只取第一个角度信号进行制作数据集
    for folder in folders:
        datas = load_txt_folder(folder)
        for data in datas:
            size = data.shape
            if size[0] < 100:
                continue
            # # 求导
            # for j in range(1, size[0]):
            #     data[j, 12] = data[j, 12] - data[j - 1, 12]
            #     data[j, 13] = data[j, 13] - data[j - 1, 13]
            #     data[j, 14] = data[j, 14] - data[j - 1, 14]
            #     data[j, 15] = data[j, 15] - data[j - 1, 15]
            #
            # data[0, 12] = 0
            # 数据和标签
            for i in range(size[0] // windows_length - 3):
                x_data.append(data[i * windows_length:(i + 1) * windows_length, :].T)
                y_data.append(np.mean(data[(i + 2) * windows_length:(i + 3) * windows_length, 12]))
                y_data.append(np.mean(data[(i + 2) * windows_length:(i + 3) * windows_length, 13]))
                y_data.append(np.mean(data[(i + 2) * windows_length:(i + 3) * windows_length, 14]))
                y_data.append(np.mean(data[(i + 2) * windows_length:(i + 3) * windows_length, 15]))

    x = np.abs(np.array(x_data))

    for i in range(12):
        M = np.max(x[:, i, :])
        m = np.min(x[:, i, :])
        x[:, i, :] = (x[:, i, :] - m) / (M - m)
    print(x.shape)
    x_data = torch.from_numpy(x)

    x_data = x_data.reshape(-1, 1, 16, 50)
    y_data = np.array(y_data).reshape(-1, 4) / degree_arg
    print(y_data[:5])
    print(y_data.shape)
    y_data = torch.from_numpy(normalization_y(y_data))
    # print(y_data[:5])
    y_data = y_data.reshape(-1, 1, 4)
    torch.save(x_data, 'x_data.pth')
    torch.save(y_data, 'y_data.pth')
    print(x_data.size(), y_data.size())


class sEMGdataset(Dataset):
    def __init__(self):
        self.x_data = torch.load('x_data.pth')
        self.labels = torch.load('y_data.pth')
        self.len = len(self.labels)

    def __getitem__(self, idx):
        data = self.x_data[idx, :, :13, :]
        label = self.labels[idx, :, 0]
        return data, label

    def __len__(self):
        return self.len


# # 数据集加载
# dataset = sEMGdataset()
# validation_split = 0.3
# shuffle_dataset = False
# random_seed = 20174145
# # 拆分数据集
# dataset_size = dataset.__len__()
# indices = list(range(dataset_size))
# split = int(np.floor(validation_split * dataset_size))
# if shuffle_dataset:
#     np.random.seed(random_seed)
#     np.random.shuffle(indices)
# train_indices, val_indices = indices[split:], indices[:split]
# # Creating PT data samplers and loaders:
# train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
# valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)
#
# train_loader = DataLoader(
#     dataset=dataset, batch_size=8, shuffle=False, sampler=train_sampler
# )
# val_loader = torch.utils.data.DataLoader(
#     dataset=dataset, batch_size=8, shuffle=False, sampler=valid_sampler
# )
if __name__ == '__main__':
    get_save_tensor()

