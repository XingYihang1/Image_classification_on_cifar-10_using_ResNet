# %% 
# 导入依赖
import pandas as pd
import os
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, random_split
from model import resnet18, resnet34, resnet50, resnet101, resnet152
from utils import try_all_gpus, Cifar10, Cifar10Test
from train import train
from utils import TrainingLogger
# %%
# 数据文件地址
data_dir = "./data/cifar-10/"

# %%
# 图像增广操作
# 对训练集的增广
train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(40),
    torchvision.transforms.RandomResizedCrop(32, scale=(0.64, 1.0), ratio=(1.0, 1.0)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])  # 选用的均值和标准差是前人在训练集上计算得到的均值和标准差
])
# 测试集
test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])

# %%
def get_dataset(batch_size, valid_ratio=0.1):
    cifar10_train_valid_dataset = Cifar10(data_dir, transform=train_transforms)  # 最后用来训练的数据集，图像增广操作

    # 设置随机种子
    torch.manual_seed(42)
    train_size = int((1 - valid_ratio) * len(cifar10_train_valid_dataset))  # 训练集大小
    valid_size = len(cifar10_train_valid_dataset) - train_size  # 验证集大小

    cifar10_train_dataset, cifar10_valid_dataset = random_split(cifar10_train_valid_dataset, [train_size, valid_size])
    cifar10_test_dataset = Cifar10Test(data_dir, transform=test_transforms)  # 测试集，不需要图像增广操作
    print("训练集图片数量:",len(cifar10_train_dataset))  # 4万5千张训练图片
    print("验证集图片数量:",len(cifar10_valid_dataset))  # 5千张验证图片
    print("测试集图片数量:",len(cifar10_test_dataset))  # 30万张测试图片
    # 获取数据加载器
    train_loader, train_valid_loader = [DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True) for dataset in [cifar10_train_dataset, cifar10_train_valid_dataset]]
    # 测试集不要打乱，因为后面要按顺序提交结果
    valid_loader, test_loader = [DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False) for dataset in [cifar10_valid_dataset, cifar10_test_dataset]]
    return train_loader, train_valid_loader, valid_loader, test_loader, cifar10_train_valid_dataset, cifar10_test_dataset

# 获取网络模型
def get_net(model_name='resnet18'):
    """
    获取网络模型
    model_name: 模型名称,支持resnet18, resnet34, resnet50, resnet101, resnet152
    """
    num_classes = 10
    if model_name == 'resnet18':
        net = resnet18(num_classes, 3)  # 因为是rgb图像，所以通道数为3
    elif model_name == 'resnet34':
        net = resnet34(num_classes, 3)
    elif model_name == 'resnet50':
        net = resnet50(num_classes, 3)
    elif model_name == 'resnet101':
        net = resnet101(num_classes, 3)
    elif model_name == 'resnet152':
        net = resnet152(num_classes, 3)
    return net

# %%
def train_and_adjust_parameters(train_loader, valid_loader, model_name='resnet18'):
    """
    训练并调整参数
    train_loader: 训练数据加载器
    valid_loader: 验证数据加载器
    model_name: 模型名称,支持resnet18, resnet34, resnet50, resnet101, resnet152
    """
    # 创建日志记录器
    logger = TrainingLogger(log_dir=f'logs/train_valid')
    # 定义损失函数
    loss_fn = nn.CrossEntropyLoss(reduction='none')

    # 开始训练并调参
    devices, num_epochs, lr, wd = try_all_gpus(), 200, 0.00005, 5e-4
    lr_period, lr_decay, net = 10, 0.9, get_net(model_name= model_name)
    train(net, train_loader, valid_loader, loss_fn, num_epochs, lr, wd, devices, lr_period, lr_decay, model_name=model_name, logger=logger)
# %%
# 用调整好的参数在完整训练集上训练并保存模型
def train_and_save_model(train_valid_loader, loss_fn, num_epochs, lr, wd, devices, lr_period, lr_decay, model_name='resnet18'):
    """
    用调整好的参数在完整训练集上训练并保存模型
    train_valid_loader: 完整训练数据加载器
    loss_fn: 损失函数
    num_epochs: 训练轮数
    lr: 学习率
    wd: 权重衰减
    devices: 设备
    lr_period: 学习率衰减周期
    lr_decay: 学习率衰减率
    model_name: 模型名称,支持resnet18, resnet34, resnet50, resnet101, resnet152
    """
    # 创建日志记录器
    logger = TrainingLogger(log_dir=f'logs/train')
    net = get_net(model_name= model_name)
    train(net, train_valid_loader, None, loss_fn, num_epochs, lr, wd, devices, lr_period, lr_decay, model_name=model_name, logger=logger)
    os.makedirs('trained_models', exist_ok=True)
    torch.save(net.state_dict(), f'trained_models/{model_name}_cifar10.pth')
# %%
# 对测试集进行分类并提交结果
def apply_to_test_dataset(test_loader, cifar10_train_valid_dataset, cifar10_test_dataset, model_name='resnet18'):
    """
    对测试集进行分类并提交结果
    test_loader: 测试数据加载器
    cifar10_train_valid_dataset: 完整训练数据集
    cifar10_test_dataset: 测试数据集
    model_name: 模型名称,支持resnet18, resnet34, resnet50, resnet101, resnet152
    """
    net = get_net(model_name= model_name)
    # 将训练好的模型加载到网络中
    net.load_state_dict(torch.load(f'trained_models/{model_name}_cifar10.pth', weights_only=True))
    preds = []
    devices = try_all_gpus()
    net.to(devices[0])
    net.eval()
    with torch.no_grad():
        for X, _ in test_loader:
            y_hat = net(X.to(devices[0]))
            preds.extend(y_hat.argmax(dim=1).type(torch.int32).cpu().numpy())
    # 提交结果
    sorted_idx = list(range(1, len(cifar10_test_dataset) + 1))
    sorted_idx.sort(key=lambda x: str(x))
    df = pd.DataFrame({'id': sorted_idx, 'label': preds})
    # 因为这个preds只是预测的标签的索引,需要返回真实的标签
    df['label'] = df['label'].apply(lambda x: cifar10_train_valid_dataset.index_label_dict[x])
    os.makedirs('submission', exist_ok=True)
    # 按id排序
    df = df.sort_values(by='id')
    df.to_csv('submission/submission.csv', index=False)
# %%
def main(batch_size, is_test = False, is_validate = True, model_name='resnet18', **kwargs):
    """
    主函数
    batch_size: 批大小
    is_test: 是否需要对测试集进行分类并提交结果
    is_validate: 是否验证模型性能调餐
    model_name: 模型名称,支持resnet18, resnet34, resnet50, resnet101, resnet152
    kwargs: 其他参数
    """
    if not is_test and is_validate:
        train_loader, _ , valid_loader, _, _, _ = get_dataset(batch_size, valid_ratio=0.1)
        train_and_adjust_parameters(train_loader, valid_loader, model_name=model_name)
    elif not is_test and not is_validate:
        _, train_valid_loader, _, _, _, _ = get_dataset(batch_size)
        # 已经选择好的超参数直接传入
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        num_epochs = kwargs.get('num_epochs', 100)
        lr = kwargs.get('lr', 0.00005)
        wd = kwargs.get('wd', 5e-4)
        devices = try_all_gpus()
        lr_period = kwargs.get('lr_period', 10)
        lr_decay = kwargs.get('lr_decay', 0.9)
        train_and_save_model(train_valid_loader, loss_fn, num_epochs, lr, wd, devices, lr_period, lr_decay, model_name=model_name)
    else:
        _, _, _, test_loader, cifar10_train_valid_dataset, cifar10_test_dataset = get_dataset(batch_size)
        try:
            apply_to_test_dataset(test_loader, cifar10_train_valid_dataset, cifar10_test_dataset, model_name=model_name)
        except FileNotFoundError:
            print("Error: 模型文件不存在，请先训练模型")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            print("测试集分类完成, 已形成结果文件submission.csv")


# %%
if __name__ == "__main__":
    # 训练并调整参数resnet18模型
    # main(batch_size=128, is_test=False, is_validate=True, model_name='resnet18')
    # 训练并调整参数resnet34模型
    # main(batch_size=128, is_test=False, is_validate=True, model_name='resnet34')
    # 训练并调整参数resnet50模型
    # main(batch_size=128, is_test=False, is_validate=True, model_name='resnet50')
    # 训练并调整参数resnet101模型
    # main(batch_size=128, is_test=False, is_validate=True, model_name='resnet101')
    # 训练并调整参数resnet152模型
    # main(batch_size=128, is_test=False, is_validate=True, model_name='resnet152')

    # 经调参后的模型发现resnet18模型效果最好
    # 在完整训练集上训练并保存模型
    # main(batch_size=128, is_test=False, is_validate=False, model_name='resnet18', num_epochs=200, lr=0.00005, wd=5e-4, lr_period=10, lr_decay=0.9)

    # 用训练好的模型对测试集进行分类并提交结果
    main(batch_size=128, is_test=True, is_validate=False, model_name='resnet18')

# %%
