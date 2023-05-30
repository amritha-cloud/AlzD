from pydoc import classname
from flask import Flask, flash, request, redirect, url_for, render_template
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
from keras.models import load_model


app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/stroke')
def stroke():
    return render_template('stroke.html')


@app.route('/AD')
def AD():
    return render_template('AD.html')


@app.route('/index')
def index():
    return render_template('index.html')


# Load the model from the .h5 file
# model = tf.keras.models.load_model('Xception_val_acc_8640.h5')
model = load_model('model.h5')


@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        print('upload_image filename: ' + filename)

        # Preprocess the image
        img = tf.keras.preprocessing.image.load_img(os.path.join(
            app.config['UPLOAD_FOLDER'], filename), target_size=(225, 225))
        x = tf.keras.preprocessing.image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        # Make predictions using the loaded model
        predictions = model.predict(x)
        print("value = ", predictions)

        class_names = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']

        if predictions >= 0.4:
            predicted_class = 'ModerateDemented'
        elif predictions >= 0.3:
            predicted_class = 'MildDemented'
        elif predictions >= 0.2:
            predicted_class = 'VeryMildDemented'
        else:
            predicted_class = 'NonDemented'

        flash('Image successfully uploaded and displayed below')
        return render_template('result.html', filename=filename, predicted_class=predicted_class)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


@app.route('/ADPred', methods=['POST'])
def upload_image_AD():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        print('upload_image filename: ' + filename)

        # Preprocess the image
        img = tf.keras.preprocessing.image.load_img(os.path.join(
            app.config['UPLOAD_FOLDER'], filename), target_size=(224, 224, 3))
        x = tf.keras.preprocessing.image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        # Make predictions using the loaded model
        predictions = model.predict(x)

        class_names = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
        predicted_class = class_names[np.argmax(predictions)]

        flash('Image successfully uploaded and displayed below')
        return render_template('result.html', filename=filename, predicted_class=predicted_class)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)


if __name__ == "__main__":
    app.run(debug=True)
