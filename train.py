"""
与训练模型有关的函数
"""
from utils import accuracy
from utils import Timer, Accumulator, evaluate_accuracy_gpu
from torch import nn
import torch
from d2l import torch as d2l

# 训练一个批次的数据
def train_batch(net, X, y, loss, trainer, devices):
    """训练一个批次的数据: 使用多个GPU进行训练"""
    if isinstance(X, list):
        # 如果是列表，则将每个元素转换为GPU
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = accuracy(pred, y)
    return train_loss_sum, train_acc_sum

# 总训练函数
def train(net, train_loader, valid_loader, loss_fn, num_epochs, lr, wd, devices,
        lr_period, lr_decay, model_name='resnet18', logger = None):
        """
        net: 神经网络模型
        train_loader: 训练数据加载器
        valid_loader: 验证数据加载器
        loss_fn: 损失函数
        num_epochs: 训练轮数
        lr: 学习率
        wd: 权重衰减
        devices: 设备
        lr_period: 学习率衰减周期
        lr_decay: 学习率衰减率
        model_name: 模型名称,支持resnet18, resnet34, resnet50, resnet101, resnet152
        logger: 日志记录器
        """
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_period, gamma=lr_decay)  # 学习率衰减，每隔lr_period个epoch，学习率衰减为原来的lr_decay倍，有助于收敛
        num_batches = len(train_loader)  # 批数量
        timer = Timer()  # 计时器
        legend = ['train loss', 'train acc']   # 等会画图用的标签
        if valid_loader is not None:
            legend.append('valid acc')
        animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], legend=legend)  # 动画绘制器
        net = nn.DataParallel(net, device_ids=devices).to(devices[0])  # 利用所有能用的GPU进行训练
        # 开始训练
        # 外循环是epoch
        for epoch in range(num_epochs):
            net.train()
            metric = Accumulator(3)  # 记录损失，准确率，样本数量，用于计算平均损失和准确率
            # 内循环是batch
            for i, (features, labels) in enumerate(train_loader):
                timer.start()
                loss, acc = train_batch(net, features, labels, loss_fn, optimizer, devices)
                metric.add(loss, acc, labels.shape[0])
                timer.stop()
                if (i + 1) % (num_batches // 10) == 0 or i == num_batches - 1:  # 每5个batch画一次准确率和损失，最后一个batch也打印一次
                    print("================================================")
                    print(f"epoch {epoch + 1}, batch {i + 1}, 训练集损失 {metric[0] / metric[2]:.3f}, 训练集准确率 {metric[1] / metric[2]:.3f}, 累计耗时 {timer.sum():.1f} 秒")
                    print("================================================")
                    animator.add(epoch + (i + 1) / num_batches, (metric[0] / metric[2], metric[1] / metric[2], None))

            if valid_loader is not None:
                valid_acc = evaluate_accuracy_gpu(net, valid_loader)
                # 记录日志
                logger.log_epoch(model_name, epoch + 1, metric[0] / metric[2], metric[1] / metric[2], valid_acc)
                print("================================================")
                print(f"epoch {epoch + 1}, 验证集准确率 {valid_acc:.3f}")
                print("================================================")
                animator.add(epoch + 1, (None, None, valid_acc))
            else:
                # 记录日志
                logger.log_epoch(model_name, epoch + 1, metric[0] / metric[2], metric[1] / metric[2])
            scheduler.step()  # 更新学习率
        print("================================================")
        print("训练完成")
        print("================================================")
        measures = (f"训练集损失 :{metric[0] / metric[2]:.3f}, 训练集准确率 :{metric[1] / metric[2]:.3f}")
        if valid_loader is not None:
            measures += (f", 验证集准确率 :{valid_acc:.3f}")
        print(measures + f"{metric[2] * num_epochs / timer.sum():.1f} ，每秒处理 {metric[2] * num_epochs / timer.sum():.1f} 个样本，在 {str(devices)} 上训练")
        # 记录日志
        logger.flush(model_name)