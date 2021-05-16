"""This module implements models used for training."""

import torch
import torch.nn as nn
import torchvision.models as models
from abc import abstractmethod, ABC


class Model(nn.Module, ABC):
    """An abstract class for transfer learning model architectures."""

    def __init__(self, model: nn.Module, pretrained: bool) -> None:
        """
        Parameters
        ----------
        model : nn.Module
            An architecture from torchvision library.
        pretrained : bool
            Specifies if model's weights should be downloaded and if feature
            extraction layers should be frozen.
        """
        super(Model, self).__init__()
        if pretrained:
            for param in model.parameters():
                param.requires_grad_(False)

    @abstractmethod
    def forward(self, images: torch.Tensor) -> None:
        """This method must be overriden in the subclasses.

        Parameters
        ----------
        images : torch.Tensor
            A normalized input image tensor of the shape [N x 3 x 224 x 224]
            where N is a batch size.
        """
        pass


class VGG16(Model):
    """An implementation of VGG-16 architecture with defined feature extractor
    and custom classifier.
    """

    def __init__(self, n_classes: int, pretrained: bool = True) -> None:
        """
        Parameters
        ----------
        n_classes : int
            Number of classes in the dataset.
        pretrained : bool, optional
            Specifies if model's weights should be downloaded and if feature
            extraction layers should be frozen, by default True.
        """
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

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """A definition of how the model is going to be run, from input tensor
        to output tensor.

        Parameters
        ----------
        images : torch.Tensor
            A normalized input image tensor of the shape [N x 3 x 224 x 224]
            where N is a batch size.

        Returns
        -------
        torch.Tensor
            An output tensor with predicted values of the shape [N x n] where
            N is a batch size and n is number of classes in the dataset.
        """
        predictions = self.model(images)

        return predictions


class ResNet50(Model):
    """An implementation of ResNet-50 architecture with defined feature
    extractor and custom classifier.
    """

    def __init__(self, n_classes: int, pretrained: bool = True) -> None:
        """
        Parameters
        ----------
        n_classes : int
            Number of classes in the dataset.
        pretrained : bool, optional
            Specifies if model's weights should be downloaded and if feature
            extraction layers should be frozen. by default True.
        """
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

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """A definition of how the model is going to be run, from input tensor
        to output tensor.

        Parameters
        ----------
        images : torch.Tensor
            A normalized input image tensor of the shape [N x 3 x 224 x 224]
            where N is a batch size.

        Returns
        -------
        torch.Tensor
            An output tensor with predicted values of the shape [N x n] where
            N is a batch size and n is number of classes in the dataset.
        """
        predictions = self.model(images)

        return predictions
