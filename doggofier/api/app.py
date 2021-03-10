import torch
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, render_template, request
from ..dataset import DogsDataset
from ..models import ResNet50


app = Flask(__name__)

dataset = DogsDataset('data')

model = ResNet50(130)
model.load_state_dict(torch.load('models/resnet50.pth'))
model.eval()


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


def get_prediction(image):
    output = model(image)
    _, prediction = output.max(1)
    prediction = prediction.item()

    return prediction


def render_prediction(prediction):
    categories = dataset.get_categories()
    prediction_cat = categories[prediction]

    return prediction_cat


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file is not None:
            image = transform_image(file)
            prediction = get_prediction(image)
            prediction_cat = render_prediction(prediction)

            return render_template('predict.html', pred=prediction_cat)
