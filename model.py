# %%
from torch import nn
from torch.nn import functional as F

# ResNet18或34的一个基本的残差块，用于构建ResNet18或34的残差块组
class Residual_18_34(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        """
        一个基本的残差块：由两个卷积层和一个残差连接组成
        input_channels: 输入通道数
        num_channels: 输出通道数，即卷积层的输出通道数
        use_1x1conv: 是否使用1x1卷积，即是否使用1x1卷积来增加通道数
        strides: 步长，即卷积层的步长
        """
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

# ResNet50或更深残差网络的一个基本的残差块，用于构建残差块组,也称为一个stage。
# 即Bottleneck残差块: 由1x1卷积、3x3卷积和1x1卷积组成,训练的参数更少
class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        """
        一个基本的残差块：由两个卷积层和一个残差连接组成
        input_channels: 输入通道数
        num_channels: 输出通道数，即卷积层的输出通道数
        use_1x1conv: 是否使用1x1卷积，即是否使用1x1卷积来增加通道数
        strides: 步长，即卷积层的步长
        """
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels // 4, kernel_size=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels // 4, num_channels // 4, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(num_channels // 4, num_channels, kernel_size=1)
        if use_1x1conv:
            self.conv4 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv4 = None
        self.bn1 = nn.BatchNorm2d(num_channels // 4)
        self.bn2 = nn.BatchNorm2d(num_channels // 4)
        self.bn3 = nn.BatchNorm2d(num_channels)
        if use_1x1conv:
            self.bn4 = nn.BatchNorm2d(num_channels)
        else:
            self.bn4 = None

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = F.relu(self.bn2(self.conv2(Y)))
        Y = self.bn3(self.conv3(Y))
        if self.conv4:
            X = self.bn4(self.conv4(X))
        Y += X
        return F.relu(Y)

# %%
# ResNet18的残差块组(即一个stage)：由两个残差块组成，第一个残差块增加通道数，并减半高和宽，第二个残差块保持通道数不变
def resnet_block_18_34(input_channels, num_channels, num_residuals, first_block=False):
    """
    残差块组：由两个残差块组成，第一个残差块增加通道数，并减半高和宽，第二个残差块保持通道数不变
    input_channels: 输入通道数
    num_channels: 输出通道数
    num_residuals: 残差块数量
    first_block: 是否是第一个残差块
    """
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual_18_34(input_channels, num_channels, use_1x1conv=True, strides=2))  # 第一个残差块，增加通道数，并减半高和宽
        else:
            blk.append(Residual_18_34(num_channels, num_channels))
    return nn.Sequential(*blk)

# %%
# ResNet50或更深残差网络的残差块组(即一个stage)：由两个残差块组成，第一个残差块增加通道数，并减半高和宽，第二个残差块保持通道数不变
def resnet_block(input_channels, num_channels, num_residuals, strides=1):
    """
    残差块组：由两个残差块组成，第一个残差块增加通道数，并减半高和宽，第二个残差块保持通道数不变
    input_channels: 输入通道数
    num_channels: 输出通道数
    num_residuals: 残差块数量
    strides: 步长
    """
    blk = []
    for i in range(num_residuals):
        if i == 0:
            blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=strides))  # 第一个残差块
        else:
            blk.append(Residual(num_channels, num_channels))
    return nn.Sequential(*blk)

#%%
# 主流ResNet18架构的实现
def resnet18(num_classes, in_channels=3):
    """
    一个稍微修改过的ResNet-18模型：为了适配Cifar10的32x32图像，使用3x3卷积核并去掉max-pooling层。
    num_classes: 输出类别数
    in_channels: 输入通道数
    """
    net = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU())
    net.add_module("resnet_block1", resnet_block_18_34(64, 64, 2, first_block=True))  # 第一个残差块组
    net.add_module("resnet_block2", resnet_block_18_34(64, 128, 2))  # 第二个残差块组
    net.add_module("resnet_block3", resnet_block_18_34(128, 256, 2))  # 第三个残差块组
    net.add_module("resnet_block4", resnet_block_18_34(256, 512, 2))  # 第四个残差块组
    net.add_module("global_avg_pool", nn.AdaptiveAvgPool2d((1,1)))  # 全局平均池化层
    net.add_module("fc", nn.Sequential(nn.Flatten(),
                                       nn.Linear(512, num_classes)))  # 全连接层
    return net

#%%
# 主流ResNet34架构的实现
def resnet34(num_classes, in_channels=3):
    """
    一个稍微修改过的ResNet-34模型：为了适配Cifar10的32x32图像，使用3x3卷积核并去掉max-pooling层。
    num_classes: 输出类别数
    in_channels: 输入通道数
    """
    net = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU())
    net.add_module("resnet_block1", resnet_block_18_34(64, 64, 3, first_block=True))  # 第一个残差块组
    net.add_module("resnet_block2", resnet_block_18_34(64, 128, 4))  # 第二个残差块组
    net.add_module("resnet_block3", resnet_block_18_34(128, 256, 6))  # 第三个残差块组
    net.add_module("resnet_block4", resnet_block_18_34(256, 512, 3))  # 第四个残差块组
    net.add_module("global_avg_pool", nn.AdaptiveAvgPool2d((1,1)))  # 全局平均池化层
    net.add_module("fc", nn.Sequential(nn.Flatten(),
                                       nn.Linear(512, num_classes)))  # 全连接层
    return net

#%%
# 主流ResNet50架构的实现
def resnet50(num_classes, in_channels=3):
    """
    一个稍微修改过的ResNet-50模型：为了适配Cifar10的32x32图像，使用3x3卷积核并去掉max-pooling层。
    num_classes: 输出类别数
    in_channels: 输入通道数
    """
    net = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU())
    net.add_module("resnet_block1", resnet_block(64, 256, 3, strides=1))  # 第一个残差块组，不减半高和宽
    net.add_module("resnet_block2", resnet_block(256, 512, 4, strides=2))  # 第二个残差块组, 减半高和宽
    net.add_module("resnet_block3", resnet_block(512, 1024, 6, strides=2))  # 第三个残差块组, 减半高和宽
    net.add_module("resnet_block4", resnet_block(1024, 2048, 3, strides=2))  # 第四个残差块组, 减半高和宽
    net.add_module("global_avg_pool", nn.AdaptiveAvgPool2d((1,1)))  # 全局平均池化层
    net.add_module("fc", nn.Sequential(nn.Flatten(),
                                       nn.Linear(2048, num_classes)))  # 全连接层, 输出类别数
    return net


#%%
# 主流ResNet101架构的实现
def resnet101(num_classes, in_channels=3):
    """
    一个稍微修改过的ResNet-101模型：为了适配Cifar10的32x32图像，使用3x3卷积核并去掉max-pooling层。
    num_classes: 输出类别数
    in_channels: 输入通道数
    """
    net = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU())
    net.add_module("resnet_block1", resnet_block(64, 256, 3, strides=1))  # 第一个残差块组, 不减半高和宽
    net.add_module("resnet_block2", resnet_block(256, 512, 4, strides=2))  # 第二个残差块组, 减半高和宽
    net.add_module("resnet_block3", resnet_block(512, 1024, 23, strides=2))  # 第三个残差块组, 减半高和宽
    net.add_module("resnet_block4", resnet_block(1024, 2048, 3, strides=2))  # 第四个残差块组, 减半高和宽
    net.add_module("global_avg_pool", nn.AdaptiveAvgPool2d((1,1)))  # 全局平均池化层
    net.add_module("fc", nn.Sequential(nn.Flatten(),
                                       nn.Linear(2048, num_classes)))  # 全连接层

    return net

# %%
# 主流ResNet152架构的实现
def resnet152(num_classes, in_channels=3):
    """
    一个稍微修改过的ResNet-152模型：为了适配Cifar10的32x32图像，使用3x3卷积核并去掉max-pooling层。
    num_classes: 输出类别数
    in_channels: 输入通道数
    """
    net = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU())
    net.add_module("resnet_block1", resnet_block(64, 256, 3, strides=1))  # 第一个残差块组, 不减半高和宽
    net.add_module("resnet_block2", resnet_block(256, 512, 8, strides=2))  # 第二个残差块组, 减半高和宽
    net.add_module("resnet_block3", resnet_block(512, 1024, 36, strides=2))  # 第三个残差块组, 减半高和宽
    net.add_module("resnet_block4", resnet_block(1024, 2048, 3, strides=2))  # 第四个残差块组, 减半高和宽
    net.add_module("global_avg_pool", nn.AdaptiveAvgPool2d((1,1)))  # 全局平均池化层
    net.add_module("fc", nn.Sequential(nn.Flatten(),
                                       nn.Linear(2048, num_classes)))  # 全连接层
    return net

# %%