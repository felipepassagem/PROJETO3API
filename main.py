from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)
model = load_model('models/base_mobilenetv2_fecahi.h5')

@app.route('/predict', methods=['POST'])
def predict():
    print("AQUIIII")
    file = request.files['image']
    img = Image.open(file).resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    print(class_idx)
    return jsonify({'class_index': int(class_idx)})


if __name__ == '__main__':
    app.run(debug=True, port=5000)