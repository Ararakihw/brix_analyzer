from flask import Flask, request, jsonify, render_template
import os
import joblib
import numpy as np
import uuid
from utils.spectrum_utils import extract_spectrum_features
from werkzeug.utils import secure_filename
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 模型路径映射
MODEL_PATHS = {
    "cherry_tomato": "model/cherry_tomato_model.pkl",
    "other": "model/other_model.pkl"
}

# 预加载所有模型
models = {}
for fruit, path in MODEL_PATHS.items():
    if os.path.exists(path):
        models[fruit] = joblib.load(path)
    else:
        print(f"警告：未找到模型 {fruit} -> {path}")

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': '请上传图片'}), 400

    fruit_type = request.form.get('fruitType')
    if not fruit_type or fruit_type not in models:
        return jsonify({'error': '无效的水果类型'}), 400

    file = request.files['image']
    filename = secure_filename(file.filename)
    file_ext = os.path.splitext(filename)[1]
    random_filename = str(uuid.uuid4()) + file_ext
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], random_filename)
    file.save(filepath)

    try:
        features = extract_spectrum_features(filepath).reshape(1, -1)
        model = models[fruit_type]
        brix = float(model.predict(features)[0])
        return jsonify({'brix': round(brix, 2)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
