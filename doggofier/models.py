import torch.nn as nn
import torchvision.models as models


resnet50 = models.resnet50(pretrained=True)
vgg16 = models.vgg16(pretrained=True)


class Model(nn.Module):
    def __init__(self, model, n_classes):
        super(Model, self).__init__()
        for param in model.parameters():
            param.requires_grad_(False)

        modules = list(model.children())[:-1]
        self.model = nn.Sequential(*modules)

    def _pre_forward(self, images):
        features = self.model(images)
        features = features.view(features.size(0), -1)

        return features


class VGG16(Model):
    def __init__(self, n_classes):
        super(VGG16, self).__init__(vgg16, n_classes)
        self.classifier = nn.Sequential(
            nn.Linear(7 * 7 * 512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, n_classes)
        )

    def forward(self, images):
        features = self._pre_forward(images)
        predictions = self.classifier(features)

        return predictions


class ResNet50(Model):
    def __init__(self, n_classes):
        super(ResNet50, self).__init__(resnet50, n_classes)
        self.classifier = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, n_classes)
        )

    def forward(self, images):
        features = self._pre_forward(images)
        predictions = self.classifier(features)

        return predictions
