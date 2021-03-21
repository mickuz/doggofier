import torch
from PIL import Image
import torchvision.transforms as transforms
from doggofier.dataset import DogsDataset
from doggofier.models import ResNet50


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


def load_model(model_path, num_classes):
    model = ResNet50(num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    return model


def get_prediction(image, model):
    output = model(image)
    _, prediction = output.max(1)
    prediction = prediction.item()

    return prediction


def render_prediction(prediction, dataset_root):
    categories = DogsDataset(dataset_root).get_categories()
    prediction_cat = categories[prediction]

    return prediction_cat
