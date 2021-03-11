import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, send_file
from ..dataset import DogsDataset
from ..models import ResNet50


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'uploads')

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


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')

    if request.method == 'POST':
        image_file = request.files['file']
        if image_file is not None:
            filename = image_file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image_file.save(filepath)

        return redirect(url_for('predict', filename=filename))


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/images/<filename>')
def images(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))


@app.route('/predict/<filename>')
def predict(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    image = transform_image(filepath)
    prediction = get_prediction(image)
    category = render_prediction(prediction)

    image_url = url_for('images', filename=filename)

    return render_template('predict.html', image_url=image_url, pred=category)
