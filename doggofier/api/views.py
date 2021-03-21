import os
from flask import Flask, render_template, request, redirect, url_for, send_file
from doggofier.api.utils import (transform_image, load_model, get_prediction,
                                 render_prediction)


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'uploads')


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
    model = load_model('models/resnet50.pth', 130)
    prediction = get_prediction(image, model)
    category = render_prediction(prediction, 'data')

    image_url = url_for('images', filename=filename)

    return render_template('predict.html', image_url=image_url, pred=category)
