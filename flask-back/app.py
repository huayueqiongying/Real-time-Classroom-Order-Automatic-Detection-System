import json
import os
from flask import Flask, request, jsonify, Response
import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import face_recognition
import dlib
from flask_cors import CORS
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    print("⚠️ onnxruntime未安装，行为检测功能将不可用")
    ONNX_AVAILABLE = False

app = Flask(__name__)
CORS(app)

# 初始化dlib模型
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('./dat/shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('./dat/dlib_face_recognition_resnet_model_v1.dat')

# 初始化YOLOv11行为检测模型
behavior_model_path = './model/best.onnx'
behavior_session = None
behavior_classes = [
    '玩手机', '弯腰', '书', '玩手机', '举手', '手机',
    '抬头', '阅读', '睡觉', '转头', '直立', '玩手机'
]

def load_behavior_model():
    """加载行为检测模型"""
    global behavior_session
    if not ONNX_AVAILABLE:
        print("❌ onnxruntime未安装，无法加载行为检测模型")
        return False
    
    try:
        behavior_session = ort.InferenceSession(behavior_model_path)
        print("✅ 行为检测模型加载成功")
        return True
    except Exception as e:
        print(f"❌ 行为检测模型加载失败: {e}")
        return False

# 加载行为检测模型
load_behavior_model()

registered_faces_file = 'registered_faces.json'

def load_registered_faces():
    if os.path.exists(registered_faces_file):
        with open(registered_faces_file, 'r') as file:
            return json.load(file)
    return {}

registered_faces = load_registered_faces()

def detect_behaviors(frame):
    if not ONNX_AVAILABLE or behavior_session is None:
        return []
    try:
        # 获取模型输入尺寸
        model_inputs = behavior_session.get_inputs()
        input_shape = model_inputs[0].shape
        try:
            input_height = int(input_shape[2])
            input_width = int(input_shape[3])
        except (ValueError, TypeError):
            input_height = 640
            input_width = 640

        original_height, original_width = frame.shape[:2]
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, (input_width, input_height))
        input_image = img_resized / 255.0
        input_image = input_image.transpose(2, 0, 1)
        input_tensor = input_image[np.newaxis, :, :, :].astype(np.float32)

        outputs = behavior_session.run(None, {model_inputs[0].name: input_tensor})
        predictions = np.squeeze(outputs[0]).T  # shape: [num_detections, 5+num_classes]

        conf_threshold = 0.1
        iou_threshold = 0.6

        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > conf_threshold, :]
        scores = scores[scores > conf_threshold]
        if predictions.shape[0] == 0:
            return []

        class_ids = np.argmax(predictions[:, 4:], axis=1)
        boxes = predictions[:, :4]

        # 中心点格式转角点格式
        boxes[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2)  # xmin
        boxes[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2)  # ymin
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]        # xmax
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]        # ymax

        # NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), scores.tolist(), conf_threshold, iou_threshold
        )

        detections = []
        nms_indices = []
        if indices is not None:
            if isinstance(indices, np.ndarray):
                nms_indices = indices.flatten()
            elif isinstance(indices, tuple) and len(indices) > 0:
                nms_indices = indices[0]
        if not isinstance(nms_indices, (list, np.ndarray)):
            nms_indices = []

        for i in nms_indices:
            i = int(i)
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]
            # 坐标还原到原图
            xmin = int(box[0] * original_width / input_width)
            ymin = int(box[1] * original_height / input_height)
            xmax = int(box[2] * original_width / input_width)
            ymax = int(box[3] * original_height / input_height)
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(original_width - 1, xmax)
            ymax = min(original_height - 1, ymax)
            width = xmax - xmin
            height = ymax - ymin
            if width > 0 and height > 0:
                detections.append({
                    'class': behavior_classes[int(class_id)] if int(class_id) < len(behavior_classes) else f'Class {int(class_id)}',
                    'confidence': float(score),
                    'bbox': [xmin, ymin, xmax, ymax]
                })
        # 移除所有调试输出
        # print("本帧检测到行为数：", len(detections), detections)
        return detections
    except Exception as e:
        print(f"行为检测出错: {e}")
        return []

import numpy as np

def nms(detections, iou_threshold=0.5):
    """
    detections: [{'bbox': [x1, y1, x2, y2], 'confidence': float, ...}, ...]
    返回去重后的detections
    """
    if not detections:
        return []
    boxes = np.array([d['bbox'] for d in detections])
    scores = np.array([d['confidence'] for d in detections])
    indices = np.argsort(scores)[::-1]

    keep = []
    while len(indices) > 0:
        current = indices[0]
        keep.append(current)
        if len(indices) == 1:
            break
        rest = indices[1:]
        ious = compute_iou(boxes[current], boxes[rest])
        indices = rest[ious < iou_threshold]
    return [detections[i] for i in keep]

def compute_iou(box, boxes):
    """计算一个box与一组boxes的IoU"""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - inter_area
    iou = inter_area / (union_area + 1e-6)
    return iou

def draw_chinese_boxes(frame, behaviors, font_path):
    from PIL import ImageFont, ImageDraw, Image
    import numpy as np
    import cv2

    # 转为PIL格式
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype(font_path, 20)
        # 移除所有调试输出
        # print("字体加载成功", font_path)
    except Exception as e:
        print("字体加载失败：", e)
        font = ImageFont.load_default()

    good_behaviors = {"举手", "抬头", "阅读", "直立"}
    neutral_behaviors = {"弯腰", "转头"}
    bad_behaviors = {"玩手机", "睡觉"}

    for det in behaviors:
        x1, y1, x2, y2 = det['bbox']
        label = f"{det['class']} {det['confidence']:.2f}"
        # 框颜色与下方卡片一致
        if det['class'] in good_behaviors:
            color = (0, 200, 0)      # 绿色
        elif det['class'] in neutral_behaviors:
            color = (255, 200, 0)    # 黄色
        elif det['class'] in bad_behaviors:
            color = (220, 0, 0)      # 红色
        else:
            color = (0, 255, 255)    # 其他默认青色
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        # 画带背景的中文标签
        try:
            text_bbox = draw.textbbox((x1, y1), label, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
        except AttributeError:
            text_width, text_height = font.getsize(label)
        draw.rectangle([x1, y1 - text_height, x1 + text_width, y1], fill=color)
        draw.text((x1, y1 - text_height), label, fill=(255,255,255), font=font)
    # PIL RGB to OpenCV BGR
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

@app.route('/face-register', methods=['POST'])
def face_register():
    data = request.get_json()
    student_id = data['student_id']
    image_data = data['image']
    
    # 解码并转换为 PIL 图像，然后转为 NumPy 数组以便 dlib 处理
    image = Image.open(BytesIO(base64.b64decode(image_data)))
    image_np = np.array(image)
    
    # 使用 detector 检测图像中的人脸，然后通过 sp 获取关键点，facerec 生成人脸编码。
    faces = detector(image_np, 1)
    if len(faces) != 1:
        return jsonify({'error': 'No face or multiple faces detected'}), 400
    
    shape = sp(image_np, faces[0])
    face_descriptor = facerec.compute_face_descriptor(image_np, shape)
    face_descriptor_list = [x for x in face_descriptor]
    
    # 人脸编码存储在 JSON 文件中，与学生 ID 关联
    registered_faces[student_id] = face_descriptor_list
    with open(registered_faces_file, 'w') as file:
        json.dump(registered_faces, file)
    
    return jsonify({'message': 'Face registered successfully'})

def gen_frames(stream_url, mode='face'):
    """
    生成视频帧
    mode: 'face' - 人脸识别模式, 'behavior' - 行为检测模式
    """
    print(f"正在连接视频流: {stream_url} (模式: {mode})")
    cap = cv2.VideoCapture(stream_url)
    
    # 检查视频流是否成功打开
    if not cap.isOpened():
        print(f"无法打开视频流: {stream_url}")
        # 返回错误图像
        error_frame = np.zeros((480, 480, 3), dtype=np.uint8)
        cv2.putText(error_frame, "无法连接视频流", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        ret, buffer = cv2.imencode('.jpg', error_frame)
        error_frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + error_frame_bytes + b'\r\n')
        return
    
    print(f"视频流连接成功: {stream_url}")
    
    # 设置分辨率为320*320
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)
    frame_skip = 5  # 跳过的帧数，约5fps
    frame_count = 0
    
    while True:
        success, frame = cap.read()
        if not success:
            print(f"读取视频帧失败，尝试重新连接...")
            # 尝试重新连接
            cap.release()
            cap = cv2.VideoCapture(stream_url)
            if not cap.isOpened():
                print(f"重新连接失败")
                break
            continue
        
        if frame_count % frame_skip == 0:
            try:
                if mode == 'face':
                    # 人脸识别模式
                    # 1. 转换为灰度图像
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    # 2. 检测人脸
                    faces = detector(gray, 1)
                    
                    for face in faces:
                        # 获取人脸关键点
                        shape = sp(frame, face)
                        # 计算人脸的128维编码
                        face_encoding = np.array(facerec.compute_face_descriptor(frame, shape))
                        # 比较捕获的人脸与已注册人脸库中的编码，以判断是否为已知人脸
                        matches = face_recognition.compare_faces(list(registered_faces.values()), face_encoding, tolerance=0.4)
                        
                        name = "Stranger"
                        color = (0, 0, 255)  # 默认红色标记陌生人
                        
                        if True in matches:
                            first_match_index = matches.index(True)
                            student_id = list(registered_faces.keys())[first_match_index]
                            name = student_id
                            color = (0, 255, 0)  # 绿色标记已注册人脸
                        
                        # 在人脸周围绘制矩形框
                        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), color, 2)
                        # 添加文本标签
                        cv2.putText(frame, name, (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
                elif mode == 'behavior':
                    # 行为检测模式
                    behaviors = detect_behaviors(frame)
                    # 使用PIL+SimHei.ttf绘制中文标签
                    font_path = os.path.join(os.path.dirname(__file__), "SimHei.ttf")
                    frame = draw_chinese_boxes(frame, behaviors, font_path)
                # 编码图像
                ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # 构造响应体
            except Exception as e:
                print(f"处理视频帧时出错: {e}")
                continue
        
        frame_count += 1
    
    # 释放资源
    cap.release()

@app.route('/video_feed/<stream_id>')
def video_feed(stream_id):
    try:
        # 确保stream_id是数字
        if not stream_id.isdigit():
            return jsonify({'error': 'Invalid stream ID, must be a number'}), 400
        
        stream_url = f'rtmp://116.205.102.242:9090/live/{stream_id}'
        return Response(gen_frames(stream_url, mode='face'),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        app.logger.error(f"Failed to stream video for ID {stream_id}: {str(e)}")
        return jsonify({'error': 'Failed to process video stream'}), 500

@app.route('/behavior_feed/<stream_id>')
def behavior_feed(stream_id):
    """行为检测视频流"""
    try:
        # 确保stream_id是数字
        if not stream_id.isdigit():
            return jsonify({'error': 'Invalid stream ID, must be a number'}), 400
        
        stream_url = f'rtmp://116.205.102.242:9090/live/{stream_id}'
        return Response(gen_frames(stream_url, mode='behavior'),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        app.logger.error(f"Failed to stream behavior video for ID {stream_id}: {str(e)}")
        return jsonify({'error': 'Failed to process behavior video stream'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Face recognition service is running'})

@app.route('/test_stream/<stream_id>', methods=['GET'])
def test_stream(stream_id):
    """测试视频流连接"""
    try:
        # 确保stream_id是数字
        if not stream_id.isdigit():
            return jsonify({'error': 'Invalid stream ID, must be a number'}), 400
        
        stream_url = f'rtmp://116.205.102.242:9090/live/{stream_id}'
        cap = cv2.VideoCapture(stream_url)
        
        if cap.isOpened():
            success, frame = cap.read()
            cap.release()
            
            if success:
                return jsonify({
                    'status': 'success',
                    'message': f'视频流 {stream_url} 连接正常',
                    'frame_size': f'{frame.shape[1]}x{frame.shape[0]}'
                })
            else:
                return jsonify({
                    'status': 'error',
                    'message': f'视频流 {stream_url} 连接成功但无法读取帧'
                })
        else:
            return jsonify({
                'status': 'error',
                'message': f'无法连接视频流 {stream_url}'
            })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'测试视频流时出错: {str(e)}'
        })

@app.route('/behavior_classes', methods=['GET'])
def get_behavior_classes():
    """获取支持的行为类别"""
    return jsonify({
        'classes': behavior_classes,
        'count': len(behavior_classes)
    })

@app.route('/detect_behavior', methods=['POST'])
def detect_behavior():
    """单张图片行为检测"""
    try:
        data = request.get_json()
        image_data = data['image']
        
        # 解码图像
        image = Image.open(BytesIO(base64.b64decode(image_data)))
        image_np = np.array(image)
        
        # 检测行为
        behaviors = detect_behaviors(image_np)
        
        return jsonify({
            'detections': behaviors,
            'count': len(behaviors)
        })
    except Exception as e:
        return jsonify({'error': f'行为检测失败: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='116.205.102.242', port=5000) 