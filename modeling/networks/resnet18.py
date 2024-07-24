import torch.nn as nn
from torchvision.models import alexnet, resnet18, resnet50, efficientnet_b0, vgg16, densenet121
from torchvision.models import ResNet18_Weights, ResNet50_Weights, EfficientNet_B0_Weights, VGG16_Weights, DenseNet121_Weights

# Feature Extractors
class FeatureRESNET18(nn.Module):
    def __init__(self):
        super(FeatureRESNET18, self).__init__()
        self.net = resnet18(weights=ResNet18_Weights.DEFAULT)

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)
        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)
        return x

class FeatureRESNET50(nn.Module):
    def __init__(self):
        super(FeatureRESNET50, self).__init__()
        self.net = resnet50(weights=ResNet50_Weights.DEFAULT)

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)
        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)
        return x

class FeatureEfficientNetB0(nn.Module):
    def __init__(self):
        super(FeatureEfficientNetB0, self).__init__()
        self.net = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.features = self.net.features

    def forward(self, x):
        x = self.features(x)
        return x

class FeatureVGG16(nn.Module):
    def __init__(self):
        super(FeatureVGG16, self).__init__()
        self.net = vgg16(weights=VGG16_Weights.DEFAULT).features

    def forward(self, x):
        x = self.net(x)
        return x

class FeatureDenseNet121(nn.Module):
    def __init__(self):
        super(FeatureDenseNet121, self).__init__()
        self.net = densenet121(weights=DenseNet121_Weights.DEFAULT).features

    def forward(self, x):
        x = self.net(x)
        return x

