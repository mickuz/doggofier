import os
import json
from flask import Flask, render_template, request, redirect, url_for, send_file
from app.utils import transform_image, load_model, get_prediction


app = Flask(__name__)
if app.config['ENV'] == 'production':
    app.config.from_object('app.config.ProdConfig')
else:
    app.config.from_object('app.config.DevConfig')


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

    with open(app.config['CATEGORIES_PATH'], mode='r') as categories_file:
        categories = json.load(categories_file)

    image = transform_image(filepath)
    model = load_model(app.config['MODEL_PATH'], len(categories))
    prediction = get_prediction(image, model)
    category = categories[str(prediction)]

    image_url = url_for('images', filename=filename)

    return render_template('predict.html', image_url=image_url, pred=category)
