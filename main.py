# main.py

import os
from flask import Flask, request, jsonify, send_from_directory
from keras.models import load_model
from flask_cors import CORS
from PIL import Image
import numpy as np

# O Flask servirá arquivos da raiz do projeto
app = Flask(__name__)
CORS(app) # Permite que o frontend (mesmo domínio) chame a API

# --- Carregamento do Modelo ---
# Garanta que o caminho para o modelo está correto
model = load_model('models/base_mobilenetv2_fecahi.h5')
class_names = [
    'air hockey', 'ampute football', 'archery', 'arm wrestling', 'axe throwing',
    'balance beam', 'barell racing', 'baseball', 'basketball', 'baton twirling',
    'bike polo', 'billiards', 'bmx', 'bobsled', 'bowling'
]

# --- Rota da API de Predição ---
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'Nenhuma imagem enviada.'}), 400

    file = request.files['image']
    try:
        img = Image.open(file).convert('RGB').resize((224, 224))
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

        prediction = model.predict(img_array)
        class_idx = int(np.argmax(prediction))
        class_name = class_names[class_idx]
        confidence = float(np.max(prediction))

        return jsonify({
            'class_index': class_idx,
            'class_name': class_name,
            'confidence': confidence
        })
    except Exception as e:
        # Adiciona um retorno de erro mais detalhado
        return jsonify({'error': f'Erro ao processar a imagem: {str(e)}'}), 500

# --- Rotas para servir o Frontend ---

# Rota para servir a página principal (index.html)
@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

# Rota para servir os arquivos da pasta 'images'
@app.route('/images/<path:filename>')
def serve_images(filename):
    return send_from_directory('images', filename)

# O bloco if __name__ == '__main__' é ótimo para testes locais,
# mas no Render vamos usar Gunicorn, então ele não será executado no deploy.
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)