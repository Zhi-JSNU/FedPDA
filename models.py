import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class VGG16(nn.Module):
    def __init__(self, num_classes=10, in_channels=3):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
        
        self._initialize_weights()
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class ResNet50(nn.Module):
    def __init__(self, num_classes=10, in_channels=3):
        super(ResNet50, self).__init__()
        # 加载预训练的ResNet50模型
        self.model = models.resnet50(weights=None)
        
        # 修改第一个卷积层以适应输入通道数
        if in_channels != 3:
            self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, 
                                       stride=2, padding=3, bias=False)
        
        # 修改最后的全连接层以匹配类别数
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        
        # 添加批归一化和dropout以提高性能
        self.model.bn = nn.BatchNorm1d(num_ftrs)
        self.model.dropout = nn.Dropout(0.5)
        
        # 初始化权重
        self._initialize_weights()
        
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        if hasattr(self.model, 'bn'):
            x = self.model.bn(x)
            x = self.model.dropout(x)
        x = self.model.fc(x)
        return x
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class ResNet18(nn.Module):
    def __init__(self, num_classes=10, in_channels=3):
        super(ResNet18, self).__init__()
        # 使用适合CIFAR-10的卷积参数 (较小的kernel和stride)
        self.model = models.resnet18(weights=None)
        
        # 修改第一个卷积层以适应CIFAR-10
        self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, 
                                   stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
        
        # 修改最后的全连接层以匹配类别数
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.7),  # 增加dropout比例以减轻过拟合
            nn.Linear(num_ftrs, num_classes)
        )
        
        # 添加L2正则化
        self.weight_decay = 1e-4
        
        # 初始化权重
        self._initialize_weights()
        
    def forward(self, x):
        return self.model(x)
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# 在get_model函数中添加input_size参数
def get_model(model_type, num_classes=10, in_channels=1, input_size=28):
    """获取指定类型的模型
    
    Args:
        model_type (str): 模型类型
        num_classes (int): 类别数量
        in_channels (int): 输入通道数，默认为1（灰度图像）
        input_size (int): 输入图像尺寸，默认为28（MNIST尺寸）
    
    Returns:
        nn.Module: 模型实例
    """
    if model_type == 'simple_cnn':
        return SimpleCNN(num_classes=num_classes, in_channels=in_channels, input_size=input_size)
    elif model_type == 'resnet18':
        return ResNet18(num_classes=num_classes, in_channels=in_channels)
    elif model_type == 'resnet50':
        return ResNet50(num_classes=num_classes, in_channels=in_channels)
    elif model_type == 'vgg16':
        return VGG16(num_classes=num_classes, in_channels=in_channels)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

# 找到SimpleCNN类的定义部分
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, in_channels=1, input_size=28):
        super(SimpleCNN, self).__init__()
        self.in_channels = in_channels
        self.input_size = input_size
        
        # 确保第一个卷积层的输入通道数正确
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.dropout1 = nn.Dropout2d(0.25)  # 添加2D dropout
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.dropout2 = nn.Dropout2d(0.25)  # 添加2D dropout
        
        # 计算全连接层的输入特征数
        fc_input_size = 32 * (input_size // 4) * (input_size // 4)
        
        self.fc1 = nn.Linear(fc_input_size, 128)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.5)  # 添加常规dropout
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.dropout1(x)  # 应用dropout
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.dropout2(x)  # 应用dropout
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout3(x)  # 应用dropout
        x = self.fc2(x)
        return x

def get_model_complexity(model_type):
    """获取模型复杂度评分"""
    complexity_scores = {
        'simple_cnn': 0.6,
        'resnet18': 0.8,
        'resnet50': 0.9,
        'vgg16': 0.95
    }
    return complexity_scores.get(model_type, 0.5)
