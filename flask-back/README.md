# flask-back 后端说明

本目录为“课堂学生行为检测系统”后端服务，基于 Flask 框架，支持人脸识别与学生行为检测。

## 主要功能
- 人脸注册与识别（基于 dlib）
- RTMP 视频流拉取与实时行为检测（基于 ONNX 行为模型）
- 支持多种学生行为类别的检测与可视化
- 支持中文标签显示

## 依赖环境
详见 requirements.txt，主要依赖：
- Flask / Flask-CORS
- Pillow
- numpy
- opencv-python
- face-recognition
- dlib
- onnxruntime（如需行为检测）

## 目录结构
- `app.py`：主后端服务入口，包含所有接口与推理逻辑
- `requirements.txt`：依赖包列表
- `install_dependencies.py`：一键安装依赖脚本
- `install_onnx.md`：onnxruntime安装指南
- `SimHei.ttf`：中文标签字体文件
- `registered_faces.json`：已注册人脸数据
- `../dat/`：人脸识别模型数据（如 shape_predictor_68_face_landmarks.dat 等）
- `../model/`：行为检测ONNX模型（如 best.onnx）

## 启动方式
1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   # 或 python install_dependencies.py
   ```
2. 启动服务：
   ```bash
   python app.py
   ```
   默认监听 116.205.102.242:5000

## 说明
- RTMP流地址、服务IP等已配置为公网服务器（116.205.102.242）
- 前后端可独立运行，接口联调见前端文档
- 如需自定义模型或类别，请同步修改 app.py 中相关部分

