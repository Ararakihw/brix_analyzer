from flask import Flask, request, jsonify, render_template
import os
import joblib
import numpy as np
import uuid
from utils.spectrum_utils import extract_spectrum_features
from werkzeug.utils import secure_filename
from sklearn.ensemble import RandomForestRegressor  # 引入 RandomForestRegressor

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 加载模型（确保这里加载的是训练好的随机森林回归模型）
model = joblib.load("model/random_forest_brix_model.pkl")  # 加载随机森林模型


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': '请上传图片'}), 400

    file = request.files['image']
    filename = secure_filename(file.filename)

    # 使用 uuid 生成随机文件名，并保留原始文件扩展名
    file_ext = os.path.splitext(filename)[1]  # 获取文件的扩展名
    random_filename = str(uuid.uuid4()) + file_ext  # 生成随机文件名

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], random_filename)
    file.save(filepath)

    try:
        # 提取光谱特征
        features = extract_spectrum_features(filepath).reshape(1, -1)

        # 使用训练好的随机森林模型进行预测
        brix = float(model.predict(features)[0])

        return jsonify({'brix': round(brix, 2)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
