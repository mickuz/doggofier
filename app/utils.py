import torch
from PIL import Image
import torchvision.transforms as transforms
from doggofier.models import ResNet50, VGG16


def transform_image(image_path):
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


def load_model(model, model_path, num_classes):
    if model == 'resnet50':
        model = ResNet50(num_classes, pretrained=False)
    elif model == 'vgg16':
        model = VGG16(num_classes, pretrained=False)
    else:
        raise ValueError('Wrong model!')
    model.load_state_dict(torch.load(model_path))
    model.eval()

    return model


def get_prediction(image, model):
    output = model(image)
    log_softmax, prediction = output.max(1)
    probability = torch.exp(log_softmax).item()
    prediction = prediction.item()

    return probability, prediction
