from torchvision.models import alexnet
from modeling.networks.resnet18 import FeatureRESNET18, FeatureRESNET50, FeatureEfficientNetB0, FeatureVGG16, FeatureDenseNet121 # Import the new feature extractor

NET_OUT_DIM = {
    'alexnet': 256,
    'resnet18': 512,
    'resnet50': 2048,
    'efficientnet_b0': 1280,
    'vgg16': 512,
    'densenet121': 1024
}

def build_feature_extractor(backbone, cfg):
    if backbone == "alexnet":
        print("Feature extractor: AlexNet")
        return alexnet(pretrained=True).features
    elif backbone == "resnet18":
        print("Feature extractor: ResNet-18")
        return FeatureRESNET18()
    elif backbone == "resnet50":
        print("Feature extractor: ResNet-50")
        return FeatureRESNET50()
    elif backbone == "efficientnet_b0":
        print("Feature extractor: EfficientNet-B0")
        return FeatureEfficientNetB0()
    elif backbone == "vgg16":
        print("Feature extractor: VGG16")
        return FeatureVGG16()
    elif backbone == "densenet121":
        print("Feature extractor: DenseNet121")
        return FeatureDenseNet121()
    else:
        raise NotImplementedError(f"Backbone {backbone} not implemented")
