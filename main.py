import os
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from flask_cors import CORS
from PIL import Image
import numpy as np

app = Flask(__name__)
CORS(app)

# Carrega o modelo
model = load_model('models/base_mobilenetv2_fecahi.h5')

# Nomes das classes
class_names = [
    'air hockey', 'ampute football', 'archery', 'arm wrestling', 'axe throwing',
    'balance beam', 'barell racing', 'baseball', 'basketball', 'baton twirling',
    'bike polo', 'billiards', 'bmx', 'bobsled', 'bowling'
]

# Rota de predição
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'Nenhuma imagem enviada.'}), 400

    file = request.files['image']
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

# Execução condicional
if __name__ == '__main__':
    # Checa se está no Render (Render define 'RENDER' no ambiente)
    if os.environ.get('RENDER') == 'true':
        app.run(host='0.0.0.0', port=10000)
    else:
        app.run(debug=True, port=5000)

    """
    
    Classes selecionadas (15): ['air hockey', 'ampute football', 'archery', 'arm wrestling', 'axe throwing', 'balance beam', 'barell racing', 'baseball', 'basketball', 'baton twirling', 'bike polo', 'billiards', 'bmx', 'bobsled', 'bowling']
air hockey (95%): 100%|██████████| 112/112 [00:00<00:00, 455.45it/s]

Treino: (1942, 224, 224, 3), (1942,)
Validação: (75, 224, 224, 3), (75,)

Total de classes: 15
Exemplo de codificação: air hockey -> [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0]

"""