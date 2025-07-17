import os
from flask import Flask, request, jsonify, Response, send_file
import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import cv2
import face_recognition
import dlib
from flask_cors import CORS
from scipy.spatial import distance as dist
import json
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
        # 移除所有调试输出
        # print("本帧检测到行为数：", len(detections), detections)
        return detections
    except Exception as e:
        print(f"行为检测出错: {e}")
        return []


# 在你的代码中添加以下函数

def eye_aspect_ratio(eye):
    """计算眼睛纵横比(EAR)"""
    # 计算垂直眼睛标志点之间的欧几里得距离
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # 计算水平眼睛标志点之间的欧几里得距离
    C = dist.euclidean(eye[0], eye[3])
    # 计算眼睛纵横比
    ear = (A + B) / (2.0 * C)
    return ear


def mouth_aspect_ratio(mouth):
    """计算嘴巴纵横比(MAR)"""
    # 计算垂直嘴巴标志点之间的欧几里得距离
    A = dist.euclidean(mouth[2], mouth[10])  # 51, 59
    B = dist.euclidean(mouth[4], mouth[8])  # 53, 57
    # 计算水平嘴巴标志点之间的欧几里得距离
    C = dist.euclidean(mouth[0], mouth[6])  # 49, 55
    # 计算嘴巴纵横比
    mar = (A + B) / (2.0 * C)
    return mar


def get_head_pose(shape, frame_shape):
    """计算头部姿态角度"""
    # 3D模型点
    model_points = np.array([
        (0.0, 0.0, 0.0),  # 鼻尖
        (0.0, -330.0, -65.0),  # 下巴
        (-225.0, 170.0, -135.0),  # 左眼左角
        (225.0, 170.0, -135.0),  # 右眼右角
        (-150.0, -150.0, -125.0),  # 左嘴角
        (150.0, -150.0, -125.0)  # 右嘴角
    ])

    # 2D图像点
    image_points = np.array([
        (shape[30][0], shape[30][1]),  # 鼻尖
        (shape[8][0], shape[8][1]),  # 下巴
        (shape[36][0], shape[36][1]),  # 左眼左角
        (shape[45][0], shape[45][1]),  # 右眼右角
        (shape[48][0], shape[48][1]),  # 左嘴角
        (shape[54][0], shape[54][1])  # 右嘴角
    ], dtype="double")

    # 相机内参
    height, width = frame_shape[:2]
    focal_length = width
    center = (width / 2, height / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))

    # 求解PnP问题
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    # 将旋转向量转换为旋转矩阵
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    # 计算欧拉角
    sy = np.sqrt(rotation_matrix[0, 0] * rotation_matrix[0, 0] + rotation_matrix[1, 0] * rotation_matrix[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = np.arctan2(-rotation_matrix[2, 0], sy)
        z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        y = np.arctan2(-rotation_matrix[2, 0], sy)
        z = 0

    # 转换为角度
    pitch = np.degrees(x)  # 俯仰角（点头）
    yaw = np.degrees(y)  # 偏航角（摇头）
    roll = np.degrees(z)  # 翻滚角（歪头）

    return pitch, yaw, roll


# 全局变量用于连续帧检测
sleep_frame_counters = {}  # 记录每个人的睡觉帧计数
SLEEP_CONSECUTIVE_FRAMES = 5  # 需要连续检测的帧数


def is_sleeping_pose(face_landmarks, frame_shape, person_id="default"):
    """平衡的睡觉姿势检测 - 主要依赖眼睛闭合，辅以其他特征"""
    global sleep_frame_counters

    # 定义眼睛和嘴巴的关键点索引
    LEFT_EYE_POINTS = list(range(36, 42))
    RIGHT_EYE_POINTS = list(range(42, 48))
    MOUTH_POINTS = list(range(48, 68))

    # 提取眼睛和嘴巴的关键点
    left_eye = np.array([(face_landmarks[i][0], face_landmarks[i][1]) for i in LEFT_EYE_POINTS])
    right_eye = np.array([(face_landmarks[i][0], face_landmarks[i][1]) for i in RIGHT_EYE_POINTS])
    mouth = np.array([(face_landmarks[i][0], face_landmarks[i][1]) for i in MOUTH_POINTS])

    # 计算眼睛纵横比
    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)
    ear = (left_ear + right_ear) / 2.0

    # 计算嘴巴纵横比
    mar = mouth_aspect_ratio(mouth)

    # 计算头部姿态
    pitch, yaw, roll = get_head_pose(face_landmarks, frame_shape)

    # 平衡的阈值设置
    EAR_THRESHOLD = 0.22  # 眼睛闭合阈值 - 适中的值
    MAR_THRESHOLD = 0.55  # 嘴巴张开阈值 - 适中的值
    HEAD_DOWN_THRESHOLD = 20  # 头部向下阈值 - 适中的值
    HEAD_TILT_THRESHOLD = 28  # 头部倾斜阈值 - 适中的值

    # 检测各个特征
    sleeping_indicators = []

    # 1. 眼睛闭合检测 - 核心特征
    eyes_closed = ear < EAR_THRESHOLD
    if eyes_closed:
        sleeping_indicators.append("eyes_closed")

    # 2. 头部向下低头检测 - 重要特征
    head_down = pitch > HEAD_DOWN_THRESHOLD
    if head_down:
        sleeping_indicators.append("head_down")

    # 3. 头部倾斜检测 - 辅助特征
    head_tilted = abs(roll) > HEAD_TILT_THRESHOLD
    if head_tilted:
        sleeping_indicators.append("head_tilted")

    # 4. 嘴巴微张检测 - 辅助特征
    mouth_open = mar > MAR_THRESHOLD
    if mouth_open:
        sleeping_indicators.append("mouth_open")

    # 简化的睡觉判断逻辑：
    # 方案1：眼睛闭合就认为是睡觉（最简单直接）
    # 方案2：眼睛闭合 + 任意一个辅助特征（更稳定）
    # 这里使用方案1，如果误检太多可以改为方案2

    current_sleep_detected = eyes_closed  # 方案1：只要眼睛闭合
    # current_sleep_detected = eyes_closed and len(sleeping_indicators) >= 2  # 方案2：眼睛闭合+其他特征

    # 连续帧检测逻辑
    if person_id not in sleep_frame_counters:
        sleep_frame_counters[person_id] = 0

    if current_sleep_detected:
        sleep_frame_counters[person_id] += 1
    else:
        sleep_frame_counters[person_id] = 0

    # 需要连续检测到才确认睡觉
    is_sleeping = sleep_frame_counters[person_id] >= SLEEP_CONSECUTIVE_FRAMES

    # 计算置信度
    if is_sleeping:
        # 基础置信度基于眼睛闭合程度
        base_confidence = min(0.9, max(0.6, (EAR_THRESHOLD - ear) / EAR_THRESHOLD))

        # 根据其他特征调整置信度
        if head_down:
            base_confidence += 0.1
        if head_tilted:
            base_confidence += 0.05
        if mouth_open:
            base_confidence += 0.05

        # 根据连续帧数调整置信度
        frame_confidence_bonus = min(0.1, sleep_frame_counters[person_id] * 0.02)

        confidence = min(0.95, base_confidence + frame_confidence_bonus)
    else:
        confidence = 0.0

    return is_sleeping, sleeping_indicators, {
        'ear': ear,
        'mar': mar,
        'pitch': pitch,
        'yaw': yaw,
        'roll': roll,
        'consecutive_frames': sleep_frame_counters[person_id],
        'confidence': confidence
    }

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
            color = (0, 200, 0)  # 绿色
        elif det['class'] in neutral_behaviors:
            color = (255, 200, 0)  # 黄色
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

# 删除人脸特征接口
@app.route('/face-delete/<student_id>', methods=['DELETE'])
def face_delete(student_id):
        # 加载已注册的人脸
        registered_faces = load_registered_faces()

        # 检查该ID是否存在
        if student_id in registered_faces:
            # 删除对应的人脸特征
            del registered_faces[student_id]

            # 保存更新后的数据
            with open(registered_faces_file, 'w') as file:
                json.dump(registered_faces, file)

            return {'message': 'Face deleted successfully'}
        else:
            return {'error': 'Face not found'}, 404



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
    neutral_behaviors = {"弯腰", "转头"}
    bad_behaviors = {"玩手机", "睡觉", "手机"}

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
        elif det['class'] in neutral_behaviors:
            color = (255, 200, 0)  # 黄色
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

        if frame_count % frame_skip == 0:
            try:
                if mode == 'face':
                    # 人脸识别模式
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = detector(gray, 1)

                    for face in faces:
                        shape = sp(frame, face)
                        face_encoding = np.array(facerec.compute_face_descriptor(frame, shape))
                        matches = face_recognition.compare_faces(list(registered_faces.values()), face_encoding,
                                                                 tolerance=0.4)

                        name = "Stranger"
                        color = (0, 0, 255)  # 默认红色标记陌生人

                        if True in matches:
                            first_match_index = matches.index(True)
                            student_id = list(registered_faces.keys())[first_match_index]
                            name = student_id
                            color = (0, 255, 0)  # 绿色标记已注册人脸

                        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), color, 2)
                        cv2.putText(frame, name, (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color,
                                    2)

                elif mode == 'behavior':
                    # 行为检测模式
                    behaviors = detect_behaviors(frame)
                    font_path = os.path.join(os.path.dirname(__file__), "SimHei.ttf")
                    frame = draw_chinese_boxes(frame, behaviors, font_path)
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


import uuid
from collections import deque
import sqlite3
from concurrent.futures import ThreadPoolExecutor


# 添加线程池和队列相关的全局变量
MAX_WORKERS = 4  # 根据CPU核心数调整
processing_executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
result_queues = {}  # 存储每个流的结果队列
processing_flags = {}  # 存储每个流的处理标志
# 添加异常事件相关的全局变量
event_active = {}  # 用于记录当前活跃的异常事件
event_cooldown = {}  # 用于记录事件冷却时间
cooldown_duration = 5  # 30秒冷却时间
events_db_file = 'anomaly_events.db'
video_buffer_size = 150  # 5秒 * 30fps = 150帧
stream_buffers = {}  # 存储每个流的视频缓冲区
anomaly_threshold = 0.6  # 异常行为置信度阈值
stranger_threshold = 0.6

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
    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS daily_reports
                   (
                       id
                       TEXT
                       PRIMARY
                       KEY,
                       report_date
                       TEXT
                       UNIQUE,
                       report_content
                       TEXT,
                       created_at
                       TEXT
                   )
                   ''')
    conn.commit()
    conn.close()


# 初始化事件数据库
init_events_db()

def process_frame_async(frame, stream_id):
    """平衡的异步处理单帧函数"""
    try:
        # 行为检测
        behaviors = detect_behaviors(frame)

        # 人脸识别和睡觉检测
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 1)

        registered_face_areas = []
        face_results = []
        sleep_detections = []

        for face in faces:
            shape = sp(frame, face)
            shape_np = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])

            # 先进行人脸识别以获取person_id
            face_encoding = np.array(facerec.compute_face_descriptor(frame, shape))
            matches = face_recognition.compare_faces(list(registered_faces.values()), face_encoding,
                                                     tolerance=0.4)
            face_distances = face_recognition.face_distance(list(registered_faces.values()), face_encoding)

            name = "Stranger"
            color = (0, 0, 255)
            is_registered = False
            stranger_confidence = 0.0
            person_id = "Stranger"

            if True in matches:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    student_id = list(registered_faces.keys())[best_match_index]
                    name = student_id
                    color = (0, 255, 0)
                    is_registered = True
                    person_id = student_id

                    face_bbox = [face.left(), face.top(), face.right(), face.bottom()]
                    registered_face_areas.append({
                        'student_id': student_id,
                        'bbox': face_bbox
                    })
            else:
                if len(face_distances) > 0:
                    min_distance = np.min(face_distances)
                    stranger_confidence = min(1.0, max(0.0, (min_distance - 0.4) / 0.6))
                else:
                    stranger_confidence = 1.0

            # 使用person_id进行睡觉检测
            is_sleeping, sleep_indicators, sleep_metrics = is_sleeping_pose(shape_np, frame.shape, person_id)

            # 只有确认睡觉时才添加到检测结果
            if is_sleeping:
                face_bbox = [face.left(), face.top(), face.right(), face.bottom()]
                sleep_detections.append({
                    'class': '睡觉',
                    'confidence': sleep_metrics['confidence'],
                    'bbox': face_bbox,
                    'sleep_indicators': sleep_indicators,
                    'sleep_metrics': sleep_metrics
                })

            face_results.append({
                'bbox': [face.left(), face.top(), face.right(), face.bottom()],
                'name': name,
                'color': color,
                'is_registered': is_registered,
                'stranger_confidence': stranger_confidence,
                'is_sleeping': is_sleeping,
                'sleep_indicators': sleep_indicators if is_sleeping else [],
                'sleep_metrics': sleep_metrics if is_sleeping else {}
            })

        # 将睡觉检测结果合并到行为检测结果中
        all_behaviors = behaviors + sleep_detections

        # 为行为检测结果关联用户信息
        enhanced_behaviors = []
        for behavior in all_behaviors:
            behavior_bbox = behavior['bbox']
            behavior_copy = behavior.copy()

            associated_user = None
            max_overlap = 0
            stranger_confidence = 0.0

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
                            stranger_confidence = face_result['stranger_confidence']
                            break

            if associated_user:
                behavior_copy['student_id'] = associated_user
                behavior_copy['is_registered'] = associated_user != "Stranger"
                if associated_user == "Stranger":
                    behavior_copy['stranger_confidence'] = stranger_confidence

            enhanced_behaviors.append(behavior_copy)

        # 检查异常条件
        is_anomaly, event_type, behavior_class, confidence, student_id = check_anomaly_conditions(
            enhanced_behaviors, stream_id)

        return {
            'face_results': face_results,
            'enhanced_behaviors': enhanced_behaviors,
            'is_anomaly': is_anomaly,
            'event_type': event_type,
            'behavior_class': behavior_class,
            'confidence': confidence,
            'student_id': student_id
        }

    except Exception as e:
        print(f"异步处理帧时出错: {e}")
        return None


def draw_results_on_frame(frame, results, font_path):
    """在帧上绘制处理结果"""
    try:
        if results:
            # 绘制人脸框和标签
            for face_result in results['face_results']:
                bbox = face_result['bbox']
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), face_result['color'], 2)
                cv2.putText(frame, face_result['name'], (bbox[0], bbox[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, face_result['color'], 2)

            # 绘制增强的行为检测结果
            frame = draw_enhanced_chinese_boxes(frame, results['enhanced_behaviors'], font_path)

    except Exception as e:
        print(f"绘制结果时出错: {e}")

    return frame
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
    """记录异常事件"""
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

        print(f"记录异常事件: {event_type} - {behavior_class} - {student_id}")

    except Exception as e:
        print(f"记录异常事件失败: {e}")


def check_anomaly_conditions(behaviors, stream_id):
    """平衡的异常条件检查"""
    bad_behaviors = {"玩手机", "睡觉", "手机"}
    current_time = time.time()
    current_active_keys = set()

    for behavior in behaviors:
        # 针对不同行为设置合适的阈值
        if behavior['class'] == '睡觉':
            threshold = 0.6  # 睡觉阈值适中
        elif behavior['class'] in ["玩手机", "手机"]:
            threshold = 0.5  # 玩手机阈值保持较低，确保检测敏感
        else:
            threshold = 0.6  # 其他行为默认阈值

        # 检查异常行为
        if behavior['class'] in bad_behaviors and behavior['confidence'] > threshold:
            student_id = behavior.get('student_id', 'Unknown')
            event_key = f"{stream_id}_{student_id}_{behavior['class']}"
            current_active_keys.add(event_key)

            # 检查冷却时间
            if (not event_active.get(event_key, False) and
                    (event_key not in event_cooldown or (
                            current_time - event_cooldown[event_key]) > cooldown_duration)):
                event_active[event_key] = True
                event_cooldown[event_key] = current_time
                return True, 'bad_behavior', behavior['class'], behavior['confidence'], student_id

        # 检查陌生人
        if behavior.get('student_id') == 'Stranger':
            stranger_confidence = behavior.get('stranger_confidence', 1.0)
            if stranger_confidence > stranger_threshold:
                event_key = f"{stream_id}_Stranger"
                current_active_keys.add(event_key)

                if (not event_active.get(event_key, False) and
                        (event_key not in event_cooldown or (
                                current_time - event_cooldown[event_key]) > cooldown_duration)):
                    event_active[event_key] = True
                    event_cooldown[event_key] = current_time
                    return True, 'stranger', behavior['class'], stranger_confidence, 'Stranger'

    # 标记已消失的异常行为为非活跃
    for key in list(event_active.keys()):
        if key not in current_active_keys:
            event_active[key] = False

    return False, None, None, None, None


# 清理函数，用于重置计数器（可选）
def reset_sleep_counters():
    """重置睡觉帧计数器"""
    global sleep_frame_counters
    sleep_frame_counters.clear()


# 清理长时间未见的person_id计数器（可选）
def cleanup_old_counters(current_person_ids):
    """清理长时间未见的person_id的计数器"""
    global sleep_frame_counters
    keys_to_remove = []
    for person_id in sleep_frame_counters:
        if person_id not in current_person_ids:
            keys_to_remove.append(person_id)

    for key in keys_to_remove:
        del sleep_frame_counters[key]


# 修改gen_frames函数，在combined模式中添加异常检测
def gen_frames_with_anomaly_detection(stream_url, mode='face'):
    """
    生成视频帧并检测异常事件（线程池优化版本）
    """
    print(f"正在连接视频流: {stream_url} (模式: {mode})")
    cap = cv2.VideoCapture(stream_url)

    # 从URL中提取stream_id
    stream_id = stream_url.split('/')[-1]

    # 初始化该流的缓冲区和队列
    if stream_id not in stream_buffers:
        stream_buffers[stream_id] = deque(maxlen=video_buffer_size)

    if stream_id not in result_queues:
        result_queues[stream_id] = queue.Queue(maxsize=5)  # 限制队列大小避免内存溢出

    processing_flags[stream_id] = False

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

    print(f"视频流连接成功: {stream_url}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 424)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 8)
    frame_skip = 2
    frame_count = 0

    # 用于存储最新的处理结果
    latest_results = None
    font_path = os.path.join(os.path.dirname(__file__), "SimHei.ttf")

    def process_callback(future):
        """处理完成后的回调函数"""
        try:
            result = future.result()
            if result:
                # 将结果放入队列
                if not result_queues[stream_id].full():
                    result_queues[stream_id].put(result)

                # 处理异常事件
                if result['is_anomaly']:
                    buffer_frames = list(stream_buffers[stream_id])
                    threading.Thread(target=record_anomaly_event,
                                     args=(stream_id, result['event_type'], result['behavior_class'],
                                           result['confidence'], result['student_id'], buffer_frames)).start()
        except Exception as e:
            print(f"处理回调时出错: {e}")
        finally:
            processing_flags[stream_id] = False

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

        try:
            # 获取最新的处理结果（非阻塞）
            try:
                while not result_queues[stream_id].empty():
                    latest_results = result_queues[stream_id].get_nowait()
            except queue.Empty:
                pass

            # 在当前帧上绘制最新的处理结果
            if latest_results:
                frame = draw_results_on_frame(frame, latest_results, font_path)

            # 如果当前没有在处理帧，且满足处理条件，则提交新的处理任务
            if (frame_count % frame_skip == 0 and
                    mode == 'combined' and
                    not processing_flags[stream_id]):
                processing_flags[stream_id] = True
                # 提交异步处理任务
                frame_copy = frame.copy()  # 创建帧的副本
                future = processing_executor.submit(process_frame_async, frame_copy, stream_id)
                future.add_done_callback(process_callback)

            # 编码图像
            ret, buffer = cv2.imencode('.jpg', frame,
                                       [int(cv2.IMWRITE_JPEG_QUALITY), 60,
                                        int(cv2.IMWRITE_JPEG_OPTIMIZE), 1])
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        except Exception as e:
            print(f"处理视频帧时出错: {e}")
            continue

        frame_count += 1

    # 清理资源
    cap.release()
    if stream_id in result_queues:
        del result_queues[stream_id]
    if stream_id in processing_flags:
        del processing_flags[stream_id]


# 应用关闭时清理线程池
import atexit


def cleanup():
    processing_executor.shutdown(wait=True)


atexit.register(cleanup)

# 新增API端点
@app.route('/anomaly_events', methods=['GET'])
def get_anomaly_events():
    """获取异常事件列表"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        status = request.args.get('status', 'all')

        conn = sqlite3.connect(events_db_file)
        cursor = conn.cursor()

        # 构建查询条件
        where_clause = ""
        params = []
        if status != 'all':
            where_clause = "WHERE status = ?"
            params.append(status)

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


import requests
DEEPSEEK_API_KEY = "sk-abe4c63cca9f466cb2f2dd789c3f3ffe"  # 替换为你的实际密钥
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"



# 辅助格式化函数
def format_stats(stats):
    if not stats: return "无异常事件"
    return "\n".join([f"- {etype}/{bclass}: {count}次" for etype, bclass, count in stats])


def format_pending_events(events):
    if not events:
        return "无待处理事件"

    formatted_events = []
    for bclass, sid, time in events:
        try:
            # 处理ISO时间格式 2025-07-16T09:34:12.697954
            if 'T' in time:
                # 分割日期和时间部分
                date_part, time_part = time.split('T')
                # 提取时间部分（去掉微秒）
                time_only = time_part.split('.')[0]  # 09:34:12
            else:
                # 如果是其他格式，尝试空格分割
                parts = time.split(' ')
                if len(parts) >= 2:
                    time_only = parts[1]
                else:
                    time_only = time  # 如果格式不对，直接使用原时间

            formatted_events.append(f"- {bclass} (学生:{sid} @ {time_only})")
        except Exception as e:
            # 如果时间解析失败，使用原时间
            formatted_events.append(f"- {bclass} (学生:{sid} @ {time})")

    return "\n".join(formatted_events)


# 在你的 Flask 应用中添加以下代码

@app.route('/generate_daily_report', methods=['POST'])
def generate_daily_report():
    conn = None
    try:
        # 获取请求数据
        data = request.get_json() or {}

        # 如果指定了日期，使用指定日期；否则使用前一天
        if 'date' in data and data['date']:
            try:
                # 验证日期格式
                report_date = datetime.strptime(data['date'], '%Y-%m-%d').strftime('%Y-%m-%d')
            except ValueError:
                return jsonify({'error': '日期格式错误，请使用 YYYY-MM-DD 格式'}), 400
        else:
            # 默认使用前一天
            report_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

        conn = sqlite3.connect(events_db_file)
        cursor = conn.cursor()

        # 检查是否已存在该日期的报告
        cursor.execute("SELECT id FROM daily_reports WHERE report_date=?", (report_date,))
        existing_report = cursor.fetchone()

        # 如果请求中包含 force=true，则删除已存在的报告
        if existing_report and data.get('force', False):
            cursor.execute("DELETE FROM daily_reports WHERE report_date=?", (report_date,))
            conn.commit()
        elif existing_report:
            return jsonify({'error': '该日期的报告已存在'}), 409

        # 调试：先查看数据库中所有记录的时间格式
        cursor.execute("SELECT created_at FROM anomaly_events LIMIT 5")
        sample_times = cursor.fetchall()
        print(f"数据库中的时间格式示例: {sample_times}")

        # 使用 LIKE 查询匹配指定日期的记录
        date_pattern = report_date + "%"  # 匹配 YYYY-MM-DD% 的所有记录

        # 获取指定日期的统计数据
        cursor.execute('''
                       SELECT event_type, behavior_class, COUNT(*) as count
                       FROM anomaly_events
                       WHERE created_at LIKE ?
                       GROUP BY event_type, behavior_class
                       ORDER BY count DESC
                       ''', (date_pattern,))
        stats = cursor.fetchall()

        print(f"查询到的统计数据: {stats}")  # 调试输出

        # 获取指定日期的未处理事件
        cursor.execute('''
                       SELECT behavior_class, student_id, timestamp
                       FROM anomaly_events
                       WHERE status='pending' AND created_at LIKE ?
                       ORDER BY created_at DESC
                           LIMIT 5
                       ''', (date_pattern,))
        pending_events = cursor.fetchall()

        print(f"待处理事件: {pending_events}")  # 调试输出

        # 如果没有查到数据，尝试其他格式
        if not stats:
            # 尝试斜杠格式
            alt_date_pattern = report_date.replace('-', '/') + "%"
            print(f"尝试斜杠格式: {alt_date_pattern}")

            cursor.execute('''
                           SELECT event_type, behavior_class, COUNT(*) as count
                           FROM anomaly_events
                           WHERE created_at LIKE ?
                           GROUP BY event_type, behavior_class
                           ORDER BY count DESC
                           ''', (alt_date_pattern,))
            stats = cursor.fetchall()

            cursor.execute('''
                           SELECT behavior_class, student_id, timestamp
                           FROM anomaly_events
                           WHERE status='pending' AND created_at LIKE ?
                           ORDER BY created_at DESC
                               LIMIT 5
                           ''', (alt_date_pattern,))
            pending_events = cursor.fetchall()

        # 计算总事件数
        total_events = sum(count for _, _, count in stats)

        # 构建提示词
        prompt = f"""
[角色]
你是一个专业的教室监控系统分析师，负责生成指定日期的监控日报。

[原始数据]
统计日期：{report_date}
总事件数：{total_events}
异常事件统计：
{format_stats(stats)}
待处理事件（前5条）：
{format_pending_events(pending_events)}

[思维链]
1. 首先概述整体情况：总事件数、主要问题类型
2. 分析事件分布：按事件类型和行为分类统计
3. 重点提醒：未处理事件的数量和典型案例
4. 提出建议：基于分析给出管理建议
5. 使用专业但易懂的语言，保持报告简洁

[报告格式]
=== {report_date} 教室监控日报 ===
[概览]
<总体情况描述>

[数据分析]
<详细分析>

[待处理事件]
<重点提醒>

[管理建议]
<具体建议>

[备注]
{f"该日期共检测到 {total_events} 起异常事件" if total_events > 0 else "该日期未检测到异常事件，课堂秩序良好"}
"""

        # 调用DeepSeek API
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "你是一个专业的教室监控系统分析师"},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 800
        }

        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload)

        if response.status_code != 200:
            error_msg = response.json().get('message', '未知错误')
            return jsonify({'error': f'DeepSeek API调用失败: {error_msg}'}), 500

        response_data = response.json()

        if not response_data.get('choices'):
            return jsonify({'error': 'DeepSeek API返回的响应中没有choices字段'}), 500

        report_content = response_data["choices"][0]["message"]["content"].strip()

        # 保存报告
        report_id = str(uuid.uuid4())
        created_at = datetime.now().isoformat()
        cursor.execute('''
                       INSERT INTO daily_reports (id, report_date, report_content, created_at)
                       VALUES (?, ?, ?, ?)
                       ''', (report_id, report_date, report_content, created_at))
        conn.commit()

        return jsonify({
            'id': report_id,
            'date': report_date,
            'content': report_content,
            'debug_info': {
                'stats_count': len(stats),
                'pending_count': len(pending_events),
                'total_events': total_events,
                'query_pattern': date_pattern
            }
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'生成报告失败: {str(e)}'}), 500
    finally:
        if conn:
            conn.close()


# 添加一个新的端点来获取可用的日期范围
@app.route('/available_dates', methods=['GET'])
def get_available_dates():
    """获取有数据的日期范围，用于前端日期选择器的限制"""
    try:
        conn = sqlite3.connect(events_db_file)
        cursor = conn.cursor()

        # 获取最早和最新的事件日期
        cursor.execute('''
                       SELECT MIN(DATE (created_at)) as min_date,
                              MAX(DATE (created_at)) as max_date,
                              COUNT(*)               as total_events
                       FROM anomaly_events
                       ''')

        result = cursor.fetchone()
        conn.close()

        if result and result[0]:
            return jsonify({
                'min_date': result[0],
                'max_date': result[1],
                'total_events': result[2]
            })
        else:
            return jsonify({
                'min_date': None,
                'max_date': None,
                'total_events': 0
            })

    except Exception as e:
        return jsonify({'error': f'获取可用日期失败: {str(e)}'}), 500


# 添加一个端点来检查指定日期是否有数据
@app.route('/check_date_data', methods=['POST'])
def check_date_data():
    """检查指定日期是否有监控数据"""
    try:
        data = request.get_json()
        check_date = data.get('date')

        if not check_date:
            return jsonify({'error': '请提供日期'}), 400

        # 验证日期格式
        try:
            datetime.strptime(check_date, '%Y-%m-%d')
        except ValueError:
            return jsonify({'error': '日期格式错误'}), 400

        conn = sqlite3.connect(events_db_file)
        cursor = conn.cursor()

        # 检查该日期是否有数据
        date_pattern = check_date + "%"
        cursor.execute('''
                       SELECT COUNT(*)
                       FROM anomaly_events
                       WHERE created_at LIKE ?
                       ''', (date_pattern,))

        event_count = cursor.fetchone()[0]

        # 检查是否已有报告
        cursor.execute('''
                       SELECT COUNT(*)
                       FROM daily_reports
                       WHERE report_date = ?
                       ''', (check_date,))

        report_exists = cursor.fetchone()[0] > 0

        conn.close()

        return jsonify({
            'has_data': event_count > 0,
            'event_count': event_count,
            'report_exists': report_exists
        })

    except Exception as e:
        return jsonify({'error': f'检查日期数据失败: {str(e)}'}), 500


# 添加一个调试接口来查看数据库中的时间格式

@app.route('/daily_reports', methods=['GET'])
def get_daily_reports():
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 5, type=int)

        conn = sqlite3.connect(events_db_file)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM daily_reports")
        total = cursor.fetchone()[0]

        offset = (page - 1) * per_page
        cursor.execute('''
                       SELECT id, report_date, report_content, created_at
                       FROM daily_reports
                       ORDER BY report_date DESC LIMIT ?
                       OFFSET ?
                       ''', (per_page, offset))

        reports = []
        for row in cursor.fetchall():
            reports.append({
                'id': row[0],
                'date': row[1],
                'content': row[2],
                'created_at': row[3]
            })

        return jsonify({
            'reports': reports,
            'total': total,
            'page': page,
            'per_page': per_page
        })

    except Exception as e:
        return jsonify({'error': f'获取报告失败: {str(e)}'}), 500
    finally:
        conn.close()


@app.route('/daily_reports/<report_id>', methods=['DELETE'])
def delete_daily_report(report_id):
    """删除指定的日报"""
    try:
        conn = sqlite3.connect(events_db_file)
        cursor = conn.cursor()

        # 检查报告是否存在
        cursor.execute("SELECT id FROM daily_reports WHERE id = ?", (report_id,))
        if not cursor.fetchone():
            conn.close()
            return jsonify({'error': '报告不存在'}), 404

        # 删除报告
        cursor.execute("DELETE FROM daily_reports WHERE id = ?", (report_id,))
        conn.commit()
        conn.close()

        return jsonify({'message': '报告删除成功'})

    except Exception as e:
        return jsonify({'error': f'删除报告失败: {str(e)}'}), 500


# 多线程处理危险区域

import json
import math
from datetime import datetime, timedelta
import threading
import time
from collections import defaultdict
import queue

# 危险区域相关的全局变量
danger_zones = {}  # 存储每个摄像头的危险区域配置
danger_zone_config_file = 'danger_zones.json'
person_tracking = defaultdict(dict)  # 跟踪人员位置和状态
danger_alerts = []  # 存储告警信息
alert_cooldown = defaultdict(float)  # 告警冷却时间
# 线程池和队列相关的全局变量（新增）
MAX_WORKERS = 4
danger_processing_executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
danger_result_queues = {}  # 存储每个流的结果队列
danger_processing_flags = {}  # 存储每个流的处理标志

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
    """优化的危险区域违规检查"""
    if stream_id not in danger_zones:
        return []

    zone_config = danger_zones[stream_id]
    if not zone_config.get('enabled', False):
        return []

    zones = zone_config.get('zones', [])
    if not zones:
        print("没有配置任何危险区域")
        return []

    safety_distance = float(zone_config.get('safety_distance', DEFAULT_SAFETY_DISTANCE))
    stay_time = float(zone_config.get('stay_time', DEFAULT_STAY_TIME))

    violations = []
    current_time = time.time()

    for behavior in behaviors:
        if behavior.get('bbox'):
            person_center = get_person_center(behavior['bbox'])
            person_id = behavior.get('student_id', f"person_{int(person_center[0])}_{int(person_center[1])}")
            print(f"检测到人员: {person_id}, 位置: {person_center}")

            for zone_idx, zone in enumerate(zones):
                zone_name = zone.get('name', f'危险区域{zone_idx + 1}')
                polygon = zone.get('polygon', [])

                if not polygon:
                    continue

                distance = point_to_polygon_distance(person_center, polygon)
                print(f"人员到区域距离: {distance}")
                is_inside = distance == 0
                is_too_close = distance < safety_distance

                if is_inside or is_too_close:
                    print(f"检测到违规: 是否在内部={is_inside}, 是否过近={is_too_close}")
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

                    should_alert = False
                    alert_type = ""

                    if is_inside:
                        should_alert = True
                        alert_type = "intrusion"
                    elif is_too_close and violation_duration >= stay_time:
                        should_alert = True
                        alert_type = "proximity"
                    print(f"是否应该告警: {should_alert}, 类型: {alert_type}, 持续时间: {violation_duration}")

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
    """在视频帧上绘制危险区域"""
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

        points = np.array(polygon, np.int32)
        points = points.reshape((-1, 1, 2))

        cv2.polylines(frame, [points], True, (0, 0, 255), 3)

        overlay = frame.copy()
        cv2.fillPoly(overlay, [points], (0, 0, 255))
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        if polygon:
            center_x = sum(p[0] for p in polygon) // len(polygon)
            center_y = sum(p[1] for p in polygon) // len(polygon)

            text_size = cv2.getTextSize(zone_name, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(frame,
                          (center_x - text_size[0] // 2 - 5, center_y - text_size[1] - 5),
                          (center_x + text_size[0] // 2 + 5, center_y + 5),
                          (0, 0, 0), -1)

            cv2.putText(frame, zone_name, (center_x - text_size[0] // 2, center_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return frame




# 新增：异步处理危险区域检测的函数
def process_danger_frame_async(frame, stream_id):
    """异步处理危险区域检测"""
    try:
        # 行为检测
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

        return {
            'enhanced_behaviors': enhanced_behaviors,
            'violations': violations,
            'faces': faces
        }

    except Exception as e:
        print(f"异步处理危险区域检测时出错: {e}")
        return None


def draw_danger_results_on_frame(frame, results, stream_id, font_path):
    """在帧上绘制危险区域检测结果"""
    try:
        if results:
            # 绘制危险区域
            frame = draw_danger_zones(frame, stream_id)

            # 绘制行为检测结果
            frame = draw_enhanced_chinese_boxes(frame, results['enhanced_behaviors'], font_path)

            # 显示告警信息
            if results['violations']:
                for i, violation in enumerate(results['violations'][-3:]):  # 显示最近3条告警
                    alert_text = f"告警: {violation['zone_name']} - {violation['alert_type']}"
                    cv2.putText(frame, alert_text, (10, 30 + i * 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    except Exception as e:
        print(f"绘制危险区域结果时出错: {e}")

    return frame


# 修改后的危险区域检测函数（使用异步处理）
def gen_frames_with_danger_detection_async(stream_url, mode='danger'):
    """
    异步处理的危险区域检测视频流生成函数
    """
    print(f"正在连接视频流: {stream_url} (模式: {mode})")
    cap = cv2.VideoCapture(stream_url)

    # 从URL中提取stream_id
    stream_id = stream_url.split('/')[-1]

    # 初始化该流的缓冲区和队列
    if stream_id not in stream_buffers:
        stream_buffers[stream_id] = deque(maxlen=video_buffer_size)

    if stream_id not in danger_result_queues:
        danger_result_queues[stream_id] = queue.Queue(maxsize=5)

    danger_processing_flags[stream_id] = False

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

    print(f"视频流连接成功: {stream_url}")

    # 设置摄像头参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 424)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 8)
    frame_skip = 2  # 降低处理频率以提高流畅度
    frame_count = 0

    # 用于存储最新的处理结果
    latest_results = None
    font_path = os.path.join(os.path.dirname(__file__), "SimHei.ttf")

    def danger_process_callback(future):
        """危险区域处理完成后的回调函数"""
        try:
            result = future.result()
            if result:
                # 将结果放入队列
                if not danger_result_queues[stream_id].full():
                    danger_result_queues[stream_id].put(result)

                # 处理违规事件
                if result['violations']:
                    danger_alerts.extend(result['violations'])
                    print(f"检测到危险区域违规: {len(result['violations'])}条")

        except Exception as e:
            print(f"危险区域处理回调时出错: {e}")
        finally:
            danger_processing_flags[stream_id] = False

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

        try:
            # 获取最新的处理结果（非阻塞）
            try:
                while not danger_result_queues[stream_id].empty():
                    latest_results = danger_result_queues[stream_id].get_nowait()
            except queue.Empty:
                pass

            # 在当前帧上绘制最新的处理结果
            if latest_results:
                frame = draw_danger_results_on_frame(frame, latest_results, stream_id, font_path)
            else:
                # 如果没有结果，至少绘制危险区域
                frame = draw_danger_zones(frame, stream_id)

            # 如果当前没有在处理帧，且满足处理条件，则提交新的处理任务
            if (frame_count % frame_skip == 0 and
                    not danger_processing_flags[stream_id]):
                danger_processing_flags[stream_id] = True
                # 提交异步处理任务
                frame_copy = frame.copy()
                future = danger_processing_executor.submit(process_danger_frame_async, frame_copy, stream_id)
                future.add_done_callback(danger_process_callback)

            # 编码图像
            ret, buffer = cv2.imencode('.jpg', frame,
                                       [int(cv2.IMWRITE_JPEG_QUALITY), 60,
                                        int(cv2.IMWRITE_JPEG_OPTIMIZE), 1])
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        except Exception as e:
            print(f"处理视频帧时出错: {e}")
            continue

        frame_count += 1

    # 清理资源
    cap.release()
    if stream_id in danger_result_queues:
        del danger_result_queues[stream_id]
    if stream_id in danger_processing_flags:
        del danger_processing_flags[stream_id]


# 初始化危险区域配置
load_danger_zones()

# 应用关闭时清理线程池
import atexit


def cleanup_danger():
    danger_processing_executor.shutdown(wait=True)


atexit.register(cleanup_danger)
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
        print(f"该摄像头的告警数: {len(stream_alerts)}")
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



# 新增危险区域检测视频流端点
@app.route('/danger_feed/<stream_id>')
def danger_feed(stream_id):
    """危险区域检测视频流"""
    try:
        if not stream_id.isdigit():
            return jsonify({'error': 'Invalid stream ID, must be a number'}), 400

        stream_url = f'rtmp://116.205.102.242:9090/live/{stream_id}'
        return Response(gen_frames_with_danger_detection_async(stream_url, mode='danger'),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        app.logger.error(f"Failed to stream danger detection video for ID {stream_id}: {str(e)}")
        return jsonify({'error': 'Failed to process danger detection video stream'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
