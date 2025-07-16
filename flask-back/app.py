import json
import os
from flask import Flask, request, jsonify, Response, send_file
import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
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

        conf_threshold = 0.18
        iou_threshold = 0.5

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
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]  # xmax
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]  # ymax

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
                    'class': behavior_classes[int(class_id)] if int(class_id) < len(
                        behavior_classes) else f'Class {int(class_id)}',
                    'confidence': float(score),
                    'bbox': [xmin, ymin, xmax, ymax]
                })
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
    except Exception as e:
        print("字体加载失败：", e)
        font = ImageFont.load_default()

    good_behaviors = {"举手", "抬头", "阅读", "直立"}
    # neutral_behaviors 已归为异常行为
    bad_behaviors = {"玩手机", "睡觉", "弯腰", "转头"}

    for det in behaviors:
        x1, y1, x2, y2 = det['bbox']
        label = f"{det['class']} {det['confidence']:.2f}"
        # 框颜色与下方卡片一致
        if det['class'] in good_behaviors:
            color = (0, 200, 0)  # 绿色
        # elif det['class'] in neutral_behaviors:
        #     color = (255, 200, 0)  # 黄色
        elif det['class'] in bad_behaviors:
            color = (220, 0, 0)  # 红色
        else:
            color = (0, 255, 255)  # 其他默认青色
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        # 画带背景的中文标签
        try:
            text_bbox = draw.textbbox((x1, y1), label, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
        except AttributeError:
            text_width, text_height = font.getsize(label)
        draw.rectangle([x1, y1 - text_height, x1 + text_width, y1], fill=color)
        draw.text((x1, y1 - text_height), label, fill=(255, 255, 255), font=font)
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


def boxes_overlap(box1, box2):
    """检查两个矩形框是否重叠"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    return not (x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min)


def calculate_overlap_ratio(box1, box2):
    """计算两个矩形框的重叠比例"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # 计算交集
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    # 检查是否有交集
    if inter_x_min >= inter_x_max or inter_y_min >= inter_y_max:
        return 0.0

    # 计算交集面积
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

    # 计算两个框的面积
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    # 计算重叠比例（相对于较小的框）
    min_area = min(box1_area, box2_area)
    if min_area == 0:
        return 0.0

    return inter_area / min_area


def draw_enhanced_chinese_boxes(frame, behaviors, font_path):
    """绘制增强的中文行为检测框，包含用户关联信息"""
    from PIL import ImageFont, ImageDraw, Image
    import numpy as np
    import cv2

    # 转为PIL格式
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype(font_path, 18)
        small_font = ImageFont.truetype(font_path, 14)
    except Exception as e:
        print("字体加载失败：", e)
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    good_behaviors = {"举手", "抬头", "阅读", "直立", "书"}
    bad_behaviors = {"玩手机", "睡觉", "手机", "弯腰", "转头"}

    for det in behaviors:
        x1, y1, x2, y2 = det['bbox']

        # 构建标签文本
        behavior_text = f"{det['class']} {det['confidence']:.2f}"
        if 'student_id' in det:
            if det['student_id'] == "Stranger":
                user_text = "陌生人"
            else:
                user_text = f"用户: {det['student_id']}"
        else:
            user_text = "未关联"

        # 框颜色
        if det['class'] in good_behaviors:
            color = (0, 200, 0)  # 绿色
        # elif det['class'] in neutral_behaviors:
        #     color = (255, 200, 0)  # 黄色
        elif det['class'] in bad_behaviors:
            color = (220, 0, 0)  # 红色
        else:
            color = (0, 255, 255)  # 其他默认青色

        # 如果是已注册用户，框线更粗
        line_width = 4 if det.get('is_registered', False) else 2

        # 绘制检测框
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        # 计算文本尺寸
        try:
            behavior_bbox = draw.textbbox((x1, y1), behavior_text, font=font)
            behavior_width = behavior_bbox[2] - behavior_bbox[0]
            behavior_height = behavior_bbox[3] - behavior_bbox[1]

            user_bbox = draw.textbbox((x1, y1), user_text, font=small_font)
            user_width = user_bbox[2] - user_bbox[0]
            user_height = user_bbox[3] - user_bbox[1]
        except AttributeError:
            # 兼容旧版本PIL
            behavior_width, behavior_height = font.getsize(behavior_text)
            user_width, user_height = small_font.getsize(user_text)

        # 绘制行为标签背景和文本
        total_width = max(behavior_width, user_width)
        total_height = behavior_height + user_height + 2

        # 背景矩形
        draw.rectangle([x1, y1 - total_height, x1 + total_width, y1], fill=color)

        # 行为文本
        draw.text((x1, y1 - total_height), behavior_text, fill=(255, 255, 255), font=font)

        # 用户文本
        draw.text((x1, y1 - user_height), user_text, fill=(255, 255, 255), font=small_font)

    # PIL RGB to OpenCV BGR
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)



def gen_frames(stream_url, mode='face'):
    """
    生成视频帧
    mode: 'face' - 人脸识别模式, 'behavior' - 行为检测模式, 'combined' - 综合模式
    """
    cap = cv2.VideoCapture(stream_url)

    # 从URL中提取stream_id
    stream_id = stream_url.split('/')[-1]

    # 初始化该流的缓冲区
    if stream_id not in stream_buffers:
        stream_buffers[stream_id] = deque(maxlen=video_buffer_size)

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

    # 设置分辨率为320*320
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 添加这行，减少缓冲区
    cap.set(cv2.CAP_PROP_FPS, 15)  # 添加这行，限制帧率
    frame_skip = 3  # 跳过的帧数，约5fps
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

        # 将原始帧添加到缓冲区
        stream_buffers[stream_id].append(frame.copy())

        if frame_count % frame_skip == 0:
            try:
                behaviors = []
                face_results = []
                registered_face_areas = []
                enhanced_behaviors = []
                stranger_present = False

                if mode == 'face' or mode == 'combined':
                    # 人脸识别
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = detector(gray, 1)
                    for face in faces:
                        shape = sp(frame, face)
                        face_encoding = np.array(facerec.compute_face_descriptor(frame, shape))
                        matches = face_recognition.compare_faces(list(registered_faces.values()), face_encoding,
                                                                 tolerance=0.4)
                        face_distances = face_recognition.face_distance(list(registered_faces.values()), face_encoding)
                        name = "Stranger"
                        color = (0, 0, 255)
                        is_registered = False
                        if True in matches:
                            best_match_index = np.argmin(face_distances)
                            if matches[best_match_index]:
                                student_id = list(registered_faces.keys())[best_match_index]
                                name = student_id
                                color = (0, 255, 0)
                                is_registered = True
                                face_bbox = [face.left(), face.top(), face.right(), face.bottom()]
                                registered_face_areas.append({
                                    'student_id': student_id,
                                    'bbox': face_bbox
                                })
                        else:
                            stranger_present = True
                        face_results.append({
                            'bbox': [face.left(), face.top(), face.right(), face.bottom()],
                            'name': name,
                            'color': color,
                            'is_registered': is_registered
                        })
                        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), color, 2)
                        cv2.putText(frame, name, (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                if mode == 'behavior' or mode == 'combined':
                    # 行为检测
                    behaviors = detect_behaviors(frame)
                    font_path = os.path.join(os.path.dirname(__file__), "SimHei.ttf")
                    frame = draw_chinese_boxes(frame, behaviors, font_path)

                # 关联行为与人脸，增强行为信息
                if (mode == 'behavior' or mode == 'combined') and len(face_results) > 0:
                    for behavior in behaviors:
                        behavior_bbox = behavior['bbox']
                        behavior_copy = behavior.copy()
                        associated_user = None
                        max_overlap = 0
                        for face_area in registered_face_areas:
                            overlap_ratio = calculate_overlap_ratio(behavior_bbox, face_area['bbox'])
                            if overlap_ratio > max_overlap and overlap_ratio > 0.1:
                                max_overlap = overlap_ratio
                                associated_user = face_area['student_id']
                        if not associated_user:
                            for face_result in face_results:
                                if not face_result['is_registered']:
                                    overlap_ratio = calculate_overlap_ratio(behavior_bbox, face_result['bbox'])
                                    if overlap_ratio > 0.1:
                                        associated_user = "Stranger"
                                        break
                        if associated_user:
                            behavior_copy['student_id'] = associated_user
                            behavior_copy['is_registered'] = associated_user != "Stranger"
                        enhanced_behaviors.append(behavior_copy)
                else:
                    enhanced_behaviors = behaviors

                # 检查异常条件（所有模式都检测）
                stranger_event_key = f"{stream_id}_Stranger"
                if stranger_present:
                    if not event_active.get(stranger_event_key, False):
                        event_active[stranger_event_key] = True
                        buffer_frames = list(stream_buffers[stream_id])
                        threading.Thread(target=record_anomaly_event,
                                         args=(stream_id, 'stranger', 'stranger', 1.0, 'Stranger', buffer_frames)).start()
                else:
                    if event_active.get(stranger_event_key, False):
                        event_active[stranger_event_key] = False

                is_anomaly, event_type, behavior_class, confidence, student_id = check_anomaly_conditions(
                    enhanced_behaviors, stream_id)
                if is_anomaly:
                    buffer_frames = list(stream_buffers[stream_id])
                    threading.Thread(target=record_anomaly_event,
                                     args=(stream_id, event_type, behavior_class, confidence, student_id,
                                           buffer_frames)).start()

                # 编码图像
                ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 40 ,int(cv2.IMWRITE_JPEG_OPTIMIZE), 1  ])
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                print(f"处理视频帧时出错: {e}")
                continue
        frame_count += 1
    cap.release()




# 添加新的综合检测端点

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


# 在现有后端代码中添加以下代码段

import threading
import time
from datetime import datetime
import uuid
from collections import deque
import sqlite3

# 添加异常事件相关的全局变量
event_active = {}  # 用于记录当前活跃的异常事件
events_db_file = 'anomaly_events.db'
video_buffer_size = 150  # 5秒 * 30fps = 150帧
stream_buffers = {}  # 存储每个流的视频缓冲区
anomaly_threshold = 0.6  # 异常行为置信度阈值


# 初始化数据库
def init_events_db():
    conn = sqlite3.connect(events_db_file)
    cursor = conn.cursor()
    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS anomaly_events
                   (
                       id
                       TEXT
                       PRIMARY
                       KEY,
                       stream_id
                       TEXT,
                       event_type
                       TEXT,
                       behavior_class
                       TEXT,
                       confidence
                       REAL,
                       student_id
                       TEXT,
                       timestamp
                       TEXT,
                       video_path
                       TEXT,
                       status
                       TEXT
                       DEFAULT
                       'pending',
                       created_at
                       TEXT,
                       handled_at
                       TEXT,
                       handler
                       TEXT
                   )
                   ''')
    conn.commit()
    conn.close()


# 初始化事件数据库
init_events_db()


def save_video_clip(stream_id, frames, event_id):
    """保存视频片段"""
    try:
        video_dir = os.path.join(os.path.dirname(__file__), 'anomaly_videos')
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)

        video_path = os.path.join(video_dir, f'{event_id}.mp4')

        if frames:
            # 获取第一帧的尺寸
            height, width = frames[0].shape[:2]

            # 创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))

            # 写入所有帧
            for frame in frames:
                out.write(frame)

            out.release()
            return video_path
    except Exception as e:
        print(f"保存视频片段失败: {e}")
        return None


def record_anomaly_event(stream_id, event_type, behavior_class, confidence, student_id, frames):
    try:
        event_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        # 保存视频片段
        video_path = save_video_clip(stream_id, frames, event_id)

        # 保存事件到数据库
        conn = sqlite3.connect(events_db_file)
        cursor = conn.cursor()
        cursor.execute('''
                       INSERT INTO anomaly_events
                       (id, stream_id, event_type, behavior_class, confidence, student_id, timestamp, video_path,
                        created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ''',
                       (event_id, stream_id, event_type, behavior_class, confidence, student_id, timestamp, video_path,
                        timestamp))
        conn.commit()
        conn.close()

    except Exception as e:
        print(f"记录异常事件失败: {e}")


def check_anomaly_conditions(behaviors, stream_id):
    bad_behaviors = {"玩手机", "睡觉", "手机", "弯腰", "转头"}
    current_active_keys = set()
    for behavior in behaviors:
        if behavior['class'] in bad_behaviors and behavior['confidence'] > anomaly_threshold:
            student_id = behavior.get('student_id', 'Unknown')
            event_key = f"{stream_id}_{student_id}_{behavior['class']}"
            current_active_keys.add(event_key)
            if not event_active.get(event_key, False):
                event_active[event_key] = True
                return True, 'bad_behavior', behavior['class'], behavior['confidence'], student_id
        if behavior.get('student_id') == 'Stranger':
            event_key = f"{stream_id}_Stranger"
            current_active_keys.add(event_key)
            if not event_active.get(event_key, False):
                event_active[event_key] = True
                return True, 'stranger', behavior['class'], behavior['confidence'], 'Stranger'
    # 标记已消失的异常行为为非活跃
    for key in list(event_active.keys()):
        if key not in current_active_keys:
            event_active[key] = False
    return False, None, None, None, None


# 修改gen_frames函数，在combined模式中添加异常检测
def gen_frames_with_anomaly_detection(stream_url, mode='face'):
    """
    生成视频帧并检测异常事件
    """
    cap = cv2.VideoCapture(stream_url)

    # 从URL中提取stream_id
    stream_id = stream_url.split('/')[-1]

    # 初始化该流的缓冲区
    if stream_id not in stream_buffers:
        stream_buffers[stream_id] = deque(maxlen=video_buffer_size)

    # 检查视频流是否成功打开
    if not cap.isOpened():
        print(f"无法打开视频流: {stream_url}")
        error_frame = np.zeros((480, 480, 3), dtype=np.uint8)
        cv2.putText(error_frame, "无法连接视频流", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        ret, buffer = cv2.imencode('.jpg', error_frame)
        error_frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + error_frame_bytes + b'\r\n')
        return

    # 设置分辨率为640*480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 添加这行，减少缓冲区
    cap.set(cv2.CAP_PROP_FPS, 15)  # 添加这行，限制帧率
    frame_skip = 2
    frame_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            print(f"读取视频帧失败，尝试重新连接...")
            cap.release()
            cap = cv2.VideoCapture(stream_url)
            if not cap.isOpened():
                print(f"重新连接失败")
                break
            continue

        # 将原始帧添加到缓冲区
        stream_buffers[stream_id].append(frame.copy())

        if frame_count % frame_skip == 0:
            try:
                if mode == 'combined':
                    # 综合模式：同时进行人脸识别和行为检测
                    behaviors = detect_behaviors(frame)

                    # 人脸识别
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = detector(gray, 1)

                    registered_face_areas = []
                    face_results = []

                    for face in faces:
                        shape = sp(frame, face)
                        face_encoding = np.array(facerec.compute_face_descriptor(frame, shape))
                        matches = face_recognition.compare_faces(list(registered_faces.values()), face_encoding,
                                                                 tolerance=0.4)
                        face_distances = face_recognition.face_distance(list(registered_faces.values()), face_encoding)

                        name = "Stranger"
                        color = (0, 0, 255)
                        is_registered = False

                        if True in matches:
                            best_match_index = np.argmin(face_distances)
                            if matches[best_match_index]:
                                student_id = list(registered_faces.keys())[best_match_index]
                                name = student_id
                                color = (0, 255, 0)
                                is_registered = True

                                face_bbox = [face.left(), face.top(), face.right(), face.bottom()]
                                registered_face_areas.append({
                                    'student_id': student_id,
                                    'bbox': face_bbox
                                })

                        face_results.append({
                            'bbox': [face.left(), face.top(), face.right(), face.bottom()],
                            'name': name,
                            'color': color,
                            'is_registered': is_registered
                        })

                    # 为行为检测结果关联用户信息
                    enhanced_behaviors = []
                    for behavior in behaviors:
                        behavior_bbox = behavior['bbox']
                        behavior_copy = behavior.copy()

                        associated_user = None
                        max_overlap = 0

                        for face_area in registered_face_areas:
                            overlap_ratio = calculate_overlap_ratio(behavior_bbox, face_area['bbox'])
                            if overlap_ratio > max_overlap and overlap_ratio > 0.1:
                                max_overlap = overlap_ratio
                                associated_user = face_area['student_id']

                        if not associated_user:
                            for face_result in face_results:
                                if not face_result['is_registered']:
                                    overlap_ratio = calculate_overlap_ratio(behavior_bbox, face_result['bbox'])
                                    if overlap_ratio > 0.1:
                                        associated_user = "Stranger"
                                        break

                        if associated_user:
                            behavior_copy['student_id'] = associated_user
                            behavior_copy['is_registered'] = associated_user != "Stranger"

                        enhanced_behaviors.append(behavior_copy)

                    # 检查异常条件
                    is_anomaly, event_type, behavior_class, confidence, student_id = check_anomaly_conditions(
                        enhanced_behaviors, stream_id)

                    if is_anomaly:
                        # 获取当前缓冲区的所有帧
                        buffer_frames = list(stream_buffers[stream_id])
                        # 在后台线程中记录异常事件
                        threading.Thread(target=record_anomaly_event,
                                         args=(stream_id, event_type, behavior_class, confidence, student_id,
                                               buffer_frames)).start()

                    # 绘制人脸框和标签
                    for face_result in face_results:
                        bbox = face_result['bbox']
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), face_result['color'], 2)
                        cv2.putText(frame, face_result['name'], (bbox[0], bbox[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, face_result['color'], 2)

                    # 绘制增强的行为检测结果
                    font_path = os.path.join(os.path.dirname(__file__), "SimHei.ttf")
                    frame = draw_enhanced_chinese_boxes(frame, enhanced_behaviors, font_path)

                # 其他模式的处理保持不变...




                # 编码图像
                ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 40,int(cv2.IMWRITE_JPEG_OPTIMIZE), 1 ])
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            except Exception as e:
                print(f"处理视频帧时出错: {e}")
                continue

        frame_count += 1

    cap.release()


# 新增API端点
@app.route('/anomaly_events', methods=['GET'])
def get_anomaly_events():
    """获取异常事件列表"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        status = request.args.get('status', 'all')
        event_type = request.args.get('event_type', 'all')

        conn = sqlite3.connect(events_db_file)
        cursor = conn.cursor()

        # 构建查询条件
        where_clauses = []
        params = []
        if status != 'all':
            where_clauses.append('status = ?')
            params.append(status)
        if event_type != 'all':
            where_clauses.append('event_type = ?')
            params.append(event_type)
        where_clause = ''
        if where_clauses:
            where_clause = 'WHERE ' + ' AND '.join(where_clauses)

        # 获取总数
        cursor.execute(f"SELECT COUNT(*) FROM anomaly_events {where_clause}", params)
        total = cursor.fetchone()[0]

        # 获取分页数据
        offset = (page - 1) * per_page
        cursor.execute(f'''
            SELECT id, stream_id, event_type, behavior_class, confidence, student_id, 
                   timestamp, video_path, status, created_at, handled_at, handler
            FROM anomaly_events {where_clause}
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
        ''', params + [per_page, offset])

        events = []
        for row in cursor.fetchall():
            events.append({
                'id': row[0],
                'stream_id': row[1],
                'event_type': row[2],
                'behavior_class': row[3],
                'confidence': row[4],
                'student_id': row[5],
                'timestamp': row[6],
                'video_path': row[7],
                'status': row[8],
                'created_at': row[9],
                'handled_at': row[10],
                'handler': row[11]
            })

        conn.close()

        return jsonify({
            'events': events,
            'total': total,
            'page': page,
            'per_page': per_page,
            'total_pages': (total + per_page - 1) // per_page
        })

    except Exception as e:
        return jsonify({'error': f'获取异常事件失败: {str(e)}'}), 500


@app.route('/anomaly_events/<event_id>/handle', methods=['POST'])
def handle_anomaly_event(event_id):
    """处理异常事件"""
    try:
        data = request.get_json()
        handler = data.get('handler', 'Unknown')

        conn = sqlite3.connect(events_db_file)
        cursor = conn.cursor()

        cursor.execute('''
                       UPDATE anomaly_events
                       SET status     = 'handled',
                           handled_at = ?,
                           handler    = ?
                       WHERE id = ?
                       ''', (datetime.now().isoformat(), handler, event_id))

        conn.commit()
        conn.close()

        return jsonify({'message': '事件处理成功'})

    except Exception as e:
        return jsonify({'error': f'处理事件失败: {str(e)}'}), 500


@app.route('/anomaly_events/<event_id>/video')
def get_anomaly_video(event_id):
    """获取异常事件视频"""
    try:
        conn = sqlite3.connect(events_db_file)
        cursor = conn.cursor()

        cursor.execute('SELECT video_path FROM anomaly_events WHERE id = ?', (event_id,))
        result = cursor.fetchone()
        conn.close()

        if result and result[0] and os.path.exists(result[0]):
            return send_file(result[0], as_attachment=True)
        else:
            return jsonify({'error': '视频文件不存在'}), 404

    except Exception as e:
        return jsonify({'error': f'获取视频失败: {str(e)}'}), 500


# 修改现有的combined_feed端点以使用新的异常检测功能
@app.route('/combined_feed/<stream_id>')
def combined_feed(stream_id):
    """综合检测视频流 - 同时进行人脸识别和行为检测"""
    try:
        if not stream_id.isdigit():
            return jsonify({'error': 'Invalid stream ID, must be a number'}), 400

        stream_url = f'rtmp://116.205.102.242:9090/live/{stream_id}'
        return Response(gen_frames_with_anomaly_detection(stream_url, mode='combined'),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        app.logger.error(f"Failed to stream combined video for ID {stream_id}: {str(e)}")
        return jsonify({'error': 'Failed to process combined video stream'}), 500


# 在现有代码基础上添加以下代码段

import json
import math
from datetime import datetime, timedelta
import threading
import time
from collections import defaultdict

# 危险区域相关的全局变量
danger_zones = {}  # 存储每个摄像头的危险区域配置
danger_zone_config_file = 'danger_zones.json'
person_tracking = defaultdict(dict)  # 跟踪人员位置和状态
danger_alerts = []  # 存储告警信息
alert_cooldown = defaultdict(float)  # 告警冷却时间

# 默认配置
DEFAULT_SAFETY_DISTANCE = 50  # 默认安全距离（像素）
DEFAULT_STAY_TIME = 3  # 默认停留时间（秒）
ALERT_COOLDOWN_TIME = 10  # 告警冷却时间（秒）


def load_danger_zones():
    """加载危险区域配置"""
    global danger_zones
    if os.path.exists(danger_zone_config_file):
        try:
            with open(danger_zone_config_file, 'r', encoding='utf-8') as file:
                danger_zones = json.load(file)
        except Exception as e:
            print(f"加载危险区域配置失败: {e}")
            danger_zones = {}
    else:
        danger_zones = {}


def save_danger_zones():
    """保存危险区域配置"""
    try:
        with open(danger_zone_config_file, 'w', encoding='utf-8') as file:
            json.dump(danger_zones, file, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存危险区域配置失败: {e}")


def point_to_polygon_distance(point, polygon):
    """计算点到多边形的最短距离"""
    if not polygon or len(polygon) < 3:
        return float('inf')

    x, y = point
    min_distance = float('inf')

    # 检查点是否在多边形内
    if point_in_polygon(point, polygon):
        return 0  # 点在多边形内，距离为0

    # 计算点到多边形各边的距离
    for i in range(len(polygon)):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % len(polygon)]
        distance = point_to_line_distance(point, p1, p2)
        min_distance = min(min_distance, distance)

    return min_distance


def point_in_polygon(point, polygon):
    """判断点是否在多边形内（射线法）"""
    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def point_to_line_distance(point, line_start, line_end):
    """计算点到线段的最短距离"""
    x, y = point
    x1, y1 = line_start
    x2, y2 = line_end

    # 线段长度的平方
    line_length_sq = (x2 - x1) ** 2 + (y2 - y1) ** 2
    if line_length_sq == 0:
        return math.sqrt((x - x1) ** 2 + (y - y1) ** 2)

    # 计算投影参数
    t = max(0, min(1, ((x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)) / line_length_sq))

    # 投影点坐标
    proj_x = x1 + t * (x2 - x1)
    proj_y = y1 + t * (y2 - y1)

    return math.sqrt((x - proj_x) ** 2 + (y - proj_y) ** 2)


def get_person_center(bbox):
    """获取人员边界框的中心点（底部中心点更合适）"""
    x1, y1, x2, y2 = bbox
    return (x1 + x2) // 2, y2  # 使用底部中心点


def check_danger_zone_violations(stream_id, behaviors):
    """检查危险区域违规 - 增强版"""
    if stream_id not in danger_zones:
        return []

    zone_config = danger_zones[stream_id]
    if not zone_config.get('enabled', False):
        return []

    zones = zone_config.get('zones', [])

    if not zones:
        return []

    # 修改为 → 添加类型转换确保是浮点数
    safety_distance = float(zone_config.get('safety_distance', DEFAULT_SAFETY_DISTANCE))
    stay_time = float(zone_config.get('stay_time', DEFAULT_STAY_TIME))

    violations = []
    current_time = time.time()

    # 扩展检测范围 - 不仅仅检测特定行为
    for behavior in behaviors:
        behavior_class = behavior.get('class', '未知')

        # 检测所有检测到的人员，不限制特定行为类别
        if behavior.get('bbox'):
            person_center = get_person_center(behavior['bbox'])
            person_id = behavior.get('student_id', f"person_{int(person_center[0])}_{int(person_center[1])}")

            for zone_idx, zone in enumerate(zones):
                zone_name = zone.get('name', f'危险区域{zone_idx + 1}')
                polygon = zone.get('polygon', [])

                if not polygon:
                    continue

                # 计算距离
                distance = point_to_polygon_distance(person_center, polygon)

                # 检查是否违规
                is_inside = distance == 0
                is_too_close = distance < safety_distance

                if is_inside or is_too_close:

                    # 更新人员跟踪信息
                    if person_id not in person_tracking[stream_id]:
                        person_tracking[stream_id][person_id] = {
                            'first_violation_time': current_time,
                            'last_position': person_center,
                            'zone_violations': {}
                        }

                    person_info = person_tracking[stream_id][person_id]
                    zone_key = f"zone_{zone_idx}"

                    if zone_key not in person_info['zone_violations']:
                        person_info['zone_violations'][zone_key] = {
                            'first_time': current_time,
                            'last_alert_time': 0
                        }

                    violation_info = person_info['zone_violations'][zone_key]
                    violation_duration = current_time - violation_info['first_time']

                    # 检查是否需要告警
                    should_alert = False
                    alert_type = ""

                    if is_inside:
                        should_alert = True
                        alert_type = "intrusion"
                    elif is_too_close and violation_duration >= stay_time:
                        should_alert = True
                        alert_type = "proximity"

                    # 检查告警冷却
                    if should_alert and (current_time - violation_info['last_alert_time']) > ALERT_COOLDOWN_TIME:
                        violation_info['last_alert_time'] = current_time

                        violation_data = {
                            'stream_id': stream_id,
                            'person_id': person_id,
                            'zone_name': zone_name,
                            'zone_index': zone_idx,
                            'alert_type': alert_type,
                            'distance': distance,
                            'safety_distance': safety_distance,
                            'violation_duration': violation_duration,
                            'person_center': person_center,
                            'timestamp': datetime.now().isoformat(),
                            'severity': 'high' if is_inside else 'medium'
                        }

                        violations.append(violation_data)

    return violations


# 修改绘制函数，添加调试信息
def draw_danger_zones(frame, stream_id):
    """在视频帧上绘制危险区域 - 增强版"""

    if stream_id not in danger_zones:
        return frame

    zone_config = danger_zones[stream_id]
    if not zone_config.get('enabled', False):
        return frame

    zones = zone_config.get('zones', [])

    for zone_idx, zone in enumerate(zones):
        polygon = zone.get('polygon', [])
        zone_name = zone.get('name', f'危险区域{zone_idx + 1}')

        if len(polygon) < 3:
            continue

        # 转换为numpy数组
        points = np.array(polygon, np.int32)
        points = points.reshape((-1, 1, 2))

        # 绘制多边形边框
        cv2.polylines(frame, [points], True, (0, 0, 255), 3)  # 红色边框，加粗

        # 绘制半透明填充
        overlay = frame.copy()
        cv2.fillPoly(overlay, [points], (0, 0, 255))
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        # 绘制区域名称
        if polygon:
            center_x = sum(p[0] for p in polygon) // len(polygon)
            center_y = sum(p[1] for p in polygon) // len(polygon)

            # 绘制文字背景
            text_size = cv2.getTextSize(zone_name, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(frame,
                          (center_x - text_size[0] // 2 - 5, center_y - text_size[1] - 5),
                          (center_x + text_size[0] // 2 + 5, center_y + 5),
                          (0, 0, 0), -1)

            cv2.putText(frame, zone_name, (center_x - text_size[0] // 2, center_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return frame


# 初始化危险区域配置
load_danger_zones()


# API端点
@app.route('/danger_zones/<stream_id>', methods=['GET'])
def get_danger_zones(stream_id):
    """获取指定摄像头的危险区域配置"""
    return jsonify(danger_zones.get(stream_id, {
        'enabled': False,
        'zones': [],
        'safety_distance': DEFAULT_SAFETY_DISTANCE,
        'stay_time': DEFAULT_STAY_TIME
    }))


@app.route('/danger_zones/<stream_id>', methods=['POST'])
def set_danger_zones(stream_id):
    """设置指定摄像头的危险区域配置"""
    try:
        data = request.get_json()
        danger_zones[stream_id] = data
        save_danger_zones()
        return jsonify({'message': '危险区域配置保存成功'})
    except Exception as e:
        return jsonify({'error': f'保存失败: {str(e)}'}), 500


@app.route('/danger_zones/<stream_id>/toggle', methods=['POST'])
def toggle_danger_zone(stream_id):
    """启用/禁用指定摄像头的危险区域检测"""
    try:
        data = request.get_json()
        enabled = data.get('enabled', False)

        if stream_id not in danger_zones:
            danger_zones[stream_id] = {
                'enabled': enabled,
                'zones': [],
                'safety_distance': DEFAULT_SAFETY_DISTANCE,
                'stay_time': DEFAULT_STAY_TIME
            }
        else:
            danger_zones[stream_id]['enabled'] = enabled

        save_danger_zones()
        return jsonify({'message': f'危险区域检测已{"启用" if enabled else "禁用"}'})
    except Exception as e:
        return jsonify({'error': f'操作失败: {str(e)}'}), 500


@app.route('/danger_alerts/<stream_id>', methods=['GET'])
def get_danger_alerts(stream_id):
    """获取指定摄像头的危险告警"""
    try:
        # 过滤该摄像头的告警
        stream_alerts = [alert for alert in danger_alerts if alert['stream_id'] == stream_id]
        return jsonify({
            'alerts': stream_alerts[-50:],  # 返回最近50条告警
            'total': len(stream_alerts)
        })
    except Exception as e:
        return jsonify({'error': f'获取告警失败: {str(e)}'}), 500


@app.route('/danger_alerts/clear/<stream_id>', methods=['POST'])
def clear_danger_alerts(stream_id):
    """清除指定摄像头的告警"""
    try:
        global danger_alerts
        danger_alerts = [alert for alert in danger_alerts if alert['stream_id'] != stream_id]
        return jsonify({'message': '告警已清除'})
    except Exception as e:
        return jsonify({'error': f'清除告警失败: {str(e)}'}), 500


# 修改gen_frames_with_anomaly_detection函数，添加危险区域检测
def gen_frames_with_danger_detection(stream_url, mode='face'):
    """
    生成视频帧并检测危险区域违规
    """
    cap = cv2.VideoCapture(stream_url)

    # 从URL中提取stream_id
    stream_id = stream_url.split('/')[-1]

    # 初始化该流的缓冲区
    if stream_id not in stream_buffers:
        stream_buffers[stream_id] = deque(maxlen=video_buffer_size)

    # 检查视频流是否成功打开
    if not cap.isOpened():
        print(f"无法打开视频流: {stream_url}")
        error_frame = np.zeros((480, 480, 3), dtype=np.uint8)
        cv2.putText(error_frame, "无法连接视频流", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        ret, buffer = cv2.imencode('.jpg', error_frame)
        error_frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + error_frame_bytes + b'\r\n')
        return

    # 设置分辨率为640*480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 15)
    frame_skip = 2
    frame_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            print(f"读取视频帧失败，尝试重新连接...")
            cap.release()
            cap = cv2.VideoCapture(stream_url)
            if not cap.isOpened():
                print(f"重新连接失败")
                break
            continue

        # 将原始帧添加到缓冲区
        stream_buffers[stream_id].append(frame.copy())

        if frame_count % frame_skip == 0:
            try:
                if mode == 'danger':
                    # 危险区域检测模式
                    behaviors = detect_behaviors(frame)

                    # 人脸识别（用于人员身份识别）
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = detector(gray, 1)

                    # 为行为检测结果关联用户信息
                    enhanced_behaviors = []
                    for behavior in behaviors:
                        behavior_bbox = behavior['bbox']
                        behavior_copy = behavior.copy()

                        # 尝试关联人脸
                        associated_user = None
                        for face in faces:
                            face_bbox = [face.left(), face.top(), face.right(), face.bottom()]
                            overlap_ratio = calculate_overlap_ratio(behavior_bbox, face_bbox)
                            if overlap_ratio > 0.1:
                                # 进行人脸识别
                                shape = sp(frame, face)
                                face_encoding = np.array(facerec.compute_face_descriptor(frame, shape))
                                matches = face_recognition.compare_faces(list(registered_faces.values()), face_encoding,
                                                                         tolerance=0.4)

                                if True in matches:
                                    best_match_index = matches.index(True)
                                    student_id = list(registered_faces.keys())[best_match_index]
                                    associated_user = student_id
                                else:
                                    associated_user = "Stranger"
                                break

                        if associated_user:
                            behavior_copy['student_id'] = associated_user

                        enhanced_behaviors.append(behavior_copy)

                    # 检查危险区域违规
                    violations = check_danger_zone_violations(stream_id, enhanced_behaviors)

                    # 如果有违规，添加到告警列表
                    if violations:
                        danger_alerts.extend(violations)

                    # 绘制危险区域
                    frame = draw_danger_zones(frame, stream_id)

                    # 绘制行为检测结果
                    font_path = os.path.join(os.path.dirname(__file__), "SimHei.ttf")
                    frame = draw_enhanced_chinese_boxes(frame, enhanced_behaviors, font_path)

                    # 在帧上显示告警信息
                    if violations:
                        for i, violation in enumerate(violations[-3:]):  # 显示最近3条告警
                            alert_text = f"告警: {violation['zone_name']} - {violation['alert_type']}"
                            cv2.putText(frame, alert_text, (10, 30 + i * 25),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # 编码图像
                ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            except Exception as e:
                print(f"处理视频帧时出错: {e}")
                continue

        frame_count += 1

    cap.release()


# 新增危险区域检测视频流端点
@app.route('/danger_feed/<stream_id>')
def danger_feed(stream_id):
    """危险区域检测视频流"""
    try:
        if not stream_id.isdigit():
            return jsonify({'error': 'Invalid stream ID, must be a number'}), 400

        stream_url = f'rtmp://116.205.102.242:9090/live/{stream_id}'
        return Response(gen_frames_with_danger_detection(stream_url, mode='danger'),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        app.logger.error(f"Failed to stream danger detection video for ID {stream_id}: {str(e)}")
        return jsonify({'error': 'Failed to process danger detection video stream'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)