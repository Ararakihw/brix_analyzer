# Brix Analyzer

基于图像光谱的糖度预测工具，支持本地模型推理与快速部署。（只是个作业，千万不要认真）

---

## 🚀 快速开始（这里提供Ubuntu）

### 1. 克隆项目

```bash
git clone https://github.com/Ararakihw/brix_analyzer.git
cd brix_analyzer
```
### 2. 创建 Python 虚拟环境(激活)
```bash
python -m venv venv
source venv/bin/activate  # Linux / macOS
```
### 3. 在虚拟环境安装依赖
```bash
pip install -r requirements.txt
```
### 4. 运行主程序
这里主程序是一个网站，需要到浏览器打开控制台输出的网址。
```bash
python main.py
```
---
### 运行指令（本人环境使用 gunicorn+Nginx）：
停止正在运行的 gunicorn 进程
```bash
pkill gunicorn
```
重启 Nginx 服务，应用新的配置或状态
```bash
sudo systemctl restart nginx
```
启动gunicorn服务器，使用4个工作进程，绑定到0.0.0.0:5000应用入口为main.py中的app对象。不知道哪里的问题需要使用虚拟环境下的，如果直接使用则会报错。
```bash
/root/venv/bin/gunicorn -w 4 -b 0.0.0.0:5000 main:app
```
采用随机森林进行训练。大部分ai完成,仅自己作业自用代码。
### 具体过程
1. [extract_spectrum_features.py](extract_spectrum_features.py)将采集到的图片提取光谱特征，以及糖度做成csv文件放到data目录下。[train](train)文件夹下，图片的名字第一个下划线前的是糖度。
2. [train.py](train.py)通过随机森林拟合光谱特征与糖度值，然后计算$R^2$
3. [main.py](main.py)程序则是开启一个网页，用户可以上传拍下的光谱图进行分析。
