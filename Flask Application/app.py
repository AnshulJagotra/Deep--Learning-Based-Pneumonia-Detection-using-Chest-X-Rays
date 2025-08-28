import os
import numpy as np
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import warnings
warnings.filterwarnings('ignore')
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image

app = Flask(__name__)

import os
# Model ko ek hi baar load karo
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../model_weights/chest_xray.h5')
model = load_model(MODEL_PATH)
print('Model loaded. Check http://127.0.0.1:5000/')

def get_className(classNo):
    if classNo == 0:
        return "Normal"
    elif classNo == 1:
        return "Pneumonia"

def getResult(img):
    img_file = image.load_img(img, target_size=(224,224))
    x = image.img_to_array(img_file)
    x = np.expand_dims(x, axis=0)
    try:
        img_data = preprocess_input(x)
    except Exception:
        img_data = x / 255.0
    classes = model.predict(img_data)
    # Agar output probability hai (e.g. [[0.8]])
    if classes.shape[-1] == 1:
        if classes[0][0] > 0.5:
            return "Normal"
        else:
            return "Pneumonia"
    # Agar output one-hot vector hai (e.g. [[1,0]] ya [[0,1]])
    else:
        class_idx = np.argmax(classes, axis=1)[0]
        return get_className(class_idx)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        result = getResult(file_path)
        return jsonify(result=result)
    return None

if __name__ == '__main__':
    app.run(debug=True)