"""
一些常用的工具函数和类
"""
import time
import numpy as np
from torch import nn
import torch
import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict

reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)
argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
size = lambda x, *args, **kwargs: x.numel(*args, **kwargs)

# 观察一下训练数据集的标签，后面重写Dataset类的时候也用得到
def read_csv_labels(fname):
    """读取fname来给标签字典返回一个文件名"""
    with open(fname, 'r') as f:
        # 跳过文件头行(列名)
        lines = f.readlines()[1:]
    tokens = [line.rstrip().split(',') for line in lines]   # rstrip() 去除字符串末尾的空白字符, split(',') 将字符串按逗号分割成列表
    return dict((name, label) for name, label in tokens)

# 创建标签索引映射
def label2index_and_index2label(data_dir):
    """
    创建标签索引映射
    返回：
    label2index_dict: 标签到索引的映射
    index2label_dict: 索引到标签的映射
    """
    all_train_labels = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))
    all_train_labels_df = pd.DataFrame(all_train_labels.items(), columns=['id', 'label'])
    labels = sorted(all_train_labels_df['label'].unique())
    label2index_dict = dict(zip(labels, range(len(labels))))
    index2label_dict = dict(zip(range(len(labels)), labels))
    return label2index_dict, index2label_dict

# 由于从kaggle上下载的cifar-10数据集并不是按照图像标签分类的文件格式，所以需要重写一下Dataset类
# 训练数据类
class Cifar10(Dataset):
    def __init__(self, data_dir, transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.train_dir = os.path.join(data_dir, 'train')
        self.img_path_list = os.listdir(self.train_dir)
        self.labels_dict = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))
        self.transform = transform
        self.label_index_dict, self.index_label_dict = label2index_and_index2label(data_dir)

    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        img_index = img_path[:-4]
        label = self.labels_dict[img_index]
        label_index = self.label_index_dict[label]
        img = Image.open(os.path.join(self.train_dir, img_path))
        if self.transform:
            img = self.transform(img)
        return img, label_index
    
    def __len__(self):
        return len(self.img_path_list)

# 测试数据类
class Cifar10Test(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        data_dir: 数据集目录
        """
        super().__init__()
        self.data_dir = data_dir
        self.test_dir = os.path.join(data_dir, 'test')
        self.img_path_list = sorted(os.listdir(self.test_dir))
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        img = Image.open(os.path.join(self.test_dir, img_path))
        if self.transform:
            img = self.transform(img)
        return img, -1  # 测试数据集没有标签，所以返回-1
    
    def __len__(self):
        return len(self.img_path_list)

# 计时器类
class Timer:
    """记录多次运行时间"""
    def __init__(self):
        """初始化计时器"""
        self.times = []
        self.start()

    def start(self):
        """开始计时"""
        self.tik = time.time()

    def stop(self):
        """停止计时并记录时间"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回总时间"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()

# 累加器类
class Accumulator:
    """累加器类: 用于累加多个变量的值"""
    def __init__(self, n):
        """初始化累加器"""
        self.data = [0.0] * n

    def add(self, *args):
        """累加多个变量的值"""
        self.data = [a + float(b) for a, b in zip(self.data, args)]  # 将多个变量的值累加到data中

    def reset(self):
        """重置累加器"""
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        """返回指定索引的值"""
        return self.data[idx]

# 计算准确率
def accuracy(y_hat, y):
    """计算准确率: 计算预测正确的样本数量

    计算准确率"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = argmax(y_hat, axis=1)
    cmp = astype(y_hat, y.dtype) == y
    return float(reduce_sum(astype(cmp, y.dtype)))

# 评估模型的准确率
def evaluate_accuracy_gpu(net, data_iter, device=None):
    """评估模型的准确率: 使用GPU评估模型的准确率"""
    if isinstance(net, nn.Module):
        net.eval()  # 设置模型为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量, 预测的总数量
    metric = Accumulator(2)

    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # 如果是列表，则将每个元素转换为GPU
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), size(y))
    return metric[0] / metric[1]

def gpu(i=0):
    """获取一个GPU设备: 如果i=0，则返回第一个GPU设备，如果i=1，则返回第二个GPU设备，以此类推"""
    return torch.device(f'cuda:{i}')

def num_gpus():
    """获取可用的GPU数量"""
    return torch.cuda.device_count()

# 获取所有可用的GPU
def try_all_gpus():
    """获取所有可用的GPU: 如果所有GPU都不可用，则返回[cpu(),]"""
    devices = [torch.device(f'cuda:{i}') for i in range(num_gpus())]
    return devices if devices else [torch.device('cpu')]

# 日志类
class TrainingLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.buffer = defaultdict(list)

    def log_epoch(self, model_name, epoch, train_loss, train_acc, valid_acc=None):
        self.buffer[model_name].append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "valid_acc": valid_acc,
        })

    def flush(self, model_name):
        if not self.buffer[model_name]:
            return
        df = pd.DataFrame(self.buffer[model_name])
        df.to_csv(os.path.join(self.log_dir, f"{model_name}.csv"), index=False)

# 比较不同模型的各种指标
def plot_metric(metric_name, model_names, log_dir="logs", save_path=None):
    plt.figure(figsize=(8, 5))
    for name in model_names:
        csv_path = os.path.join(log_dir, f"{name}.csv")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        plt.plot(df["epoch"], df[metric_name], label=name)
    plt.xlabel("epoch")
    plt.ylabel(metric_name)
    plt.legend()
    save_path = save_path or os.path.join(log_dir, f"compare_{metric_name}.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()