import torch.nn as nn
import torchvision.models as models
from abc import abstractmethod, ABC


class Model(nn.Module, ABC):
    def __init__(self, model, pretrained):
        super(Model, self).__init__()
        if pretrained:
            for param in model.parameters():
                param.requires_grad_(False)

    @abstractmethod
    def forward(self, images):
        pass


class VGG16(Model):
    def __init__(self, n_classes, pretrained=True):
        vgg16 = models.vgg16(pretrained=pretrained)

        super(VGG16, self).__init__(vgg16, pretrained)
        self.model = vgg16
        n_inputs = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Sequential(
            nn.Linear(n_inputs, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, n_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, images):
        predictions = self.model(images)

        return predictions


class ResNet50(Model):
    def __init__(self, n_classes, pretrained=True):
        resnet50 = models.resnet50(pretrained=pretrained)

        super(ResNet50, self).__init__(resnet50, pretrained)
        self.model = resnet50
        n_inputs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(n_inputs, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, n_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, images):
        predictions = self.model(images)

        return predictions
