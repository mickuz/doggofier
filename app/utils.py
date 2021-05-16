"""This module contains tools supporting the main application."""

import torch
import torchvision.transforms as transforms
from PIL import Image
from typing import Tuple
from doggofier.models import ResNet50, VGG16


def transform_image(image_path: str) -> torch.Tensor:
    """Prepares an image for inference by applying certain transforms.

    Parameters
    ----------
    image_path : str
        Path where an image is located.

    Returns
    -------
    torch.Tensor
        Image in a form of tensor ready to enter into the model.
    """
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path)
    image = transform(image)
    image.unsqueeze_(0)

    return image


def load_model(
        model_name: str,
        model_path: str,
        num_classes: int
) -> torch.nn.Module:
    """Loads the model with trained parameters for inference.

    Parameters
    ----------
    model_name : str
        Name of the model to be used for inference. It can contain only
        'resnet50' and 'vgg16' values.
    model_path : str
        A path where model state dictionary is stored.
    num_classes : int
        Number of classes in the dataset.

    Returns
    -------
    torch.nn.Module
        Model for inference.

    Raises
    ------
    ValueError
        When name of the model has an invalid value.
    """
    if model_name == 'resnet50':
        model = ResNet50(num_classes, pretrained=False)
    elif model_name == 'vgg16':
        model = VGG16(num_classes, pretrained=False)
    else:
        raise ValueError('Wrong model!')
    model.load_state_dict(torch.load(model_path))
    model.eval()

    return model


def get_prediction(
        image: torch.Tensor,
        model: torch.nn.Module
) -> Tuple[float, int]:
    """Predicts the most likely category with its associated probability.

    Parameters
    ----------
    image : torch.Tensor
        Image in a form of tensor ready to enter into the model.
    model : torch.nn.Module
        Model for inference.

    Returns
    -------
    Tuple[float, int]
        Predicted category with its probability.
    """
    output = model(image)
    log_softmax, prediction = output.max(1)
    probability = torch.exp(log_softmax).item()
    prediction = prediction.item()

    return probability, prediction
