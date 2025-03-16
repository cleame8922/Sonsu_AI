# game1.py 수정
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from PIL import ImageFont, ImageDraw, Image
import threading
import time

# 모델 및 데이터 정보
model_path = 'final.tflite'  # TFLite 모델 경로
actions = [
    '안녕하세요', '감사합니다', '사랑합니다', '어머니', '아버지', '동생', '잘', '못', '간다', '나',
    '이름', '만나다', '반갑다', '부탁', '학교', '생일', '월', '일', '나이', '고발', '복습', '학습', '눈치채다', '오다', '말', '곱다'
]  # 학습한 동작 리스트
seq_length = 30  # 모델 학습 시 사용한 시퀀스 길이

# 한글 폰트 설정
font_path = "malgun.ttf"
font = ImageFont.truetype(font_path, 30)

# 모듈 내부 상태 변수
_current_question = None
_game_result = None
_question_time = 0  # 문제가 제시된 시간
_min_confidence = 0.8  # 최소 신뢰도 임계값
_warm_up_time = 3  # 문제 제시 후 판별 시작까지의 대기 시간(초)

# 상태 접근 함수
def set_game_state(question, result):
    global _current_question, _game_result, _question_time
    _current_question = question
    _game_result = result
    _question_time = time.time()  # 문제가 새로 설정되었을 때 시간 기록

def get_game_state():
    return _current_question, _game_result

# TFLite 모델 로드
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# MediaPipe 모델 로드
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 카메라 및 시퀀스 저장 리스트
cap = None
seq = []
is_recognizing = False

def draw_text(img, text, position, font, color=(0, 255, 0)):
    """PIL을 이용해 한글을 출력하는 함수"""
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=color)
    return np.array(img_pil)

def generate_frames():
    """카메라 프레임을 스트리밍하는 함수"""
    global cap, seq, is_recognizing, _current_question, _game_result, _question_time
    
    # 카메라가 이미 열려있는지 확인
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)
    
    seq = []
    is_recognizing = True
    last_prediction_time = 0
    prediction_cooldown = 1.0  # 예측 간 최소 시간 간격(초)

    while is_recognizing:
        ret, img = cap.read()
        if not ret:
            break

        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = holistic.process(img_rgb)

        joint_list = []

        # 왼손 랜드마크 저장
        if result.left_hand_landmarks:
            for lm in result.left_hand_landmarks.landmark:
                joint_list.append([lm.x, lm.y, lm.z])
        else:
            joint_list.extend([[0, 0, 0]] * 21)

        # 오른손 랜드마크 저장
        if result.right_hand_landmarks:
            for lm in result.right_hand_landmarks.landmark:
                joint_list.append([lm.x, lm.y, lm.z])
        else:
            joint_list.extend([[0, 0, 0]] * 21)

        # 몸(상체) 랜드마크 저장
        if result.pose_landmarks:
            for lm in result.pose_landmarks.landmark:
                joint_list.append([lm.x, lm.y, lm.z])
        else:
            joint_list.extend([[0, 0, 0]] * 33)

        current_time = time.time()
        elapsed_since_question = current_time - _question_time
        ready_to_predict = elapsed_since_question >= _warm_up_time
        
        # 준비 상태 텍스트 표시
        if not ready_to_predict and _current_question:
            countdown = max(0, int(_warm_up_time - elapsed_since_question))
            img = draw_text(img, f"준비하세요... {countdown}초", (10, 50), font, (0, 0, 255))
        
        if joint_list:
            joint_list = np.array(joint_list).flatten()
            seq.append(joint_list)

            # 시퀀스 길이 유지
            if len(seq) > seq_length:
                seq.pop(0)

            # 예측 수행 (충분한 시간이 지났고, 시퀀스가 충분하고, 마지막 예측으로부터 충분히 시간이 지났을 때)
            if (ready_to_predict and 
                len(seq) == seq_length and 
                current_time - last_prediction_time >= prediction_cooldown):
                
                input_data = np.expand_dims(np.array(seq), axis=0).astype(np.float32)

                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                prediction = interpreter.get_tensor(output_details[0]['index'])[0]

                predicted_action = actions[np.argmax(prediction)]
                confidence = np.max(prediction)
                
                # 디버깅용 예측 결과 표시
                img = draw_text(img, f'예측: {predicted_action} ({confidence:.2f})', (10, 400), font, (255, 255, 255))

                # 충분한 신뢰도를 가진 경우에만 정답 판별
                if confidence >= _min_confidence:
                    color = (0, 255, 0)

                    # 정답 여부 판별
                    if _current_question:
                        if predicted_action == _current_question:
                            _game_result = "정답입니다!"
                            color = (0, 255, 0)
                        else:
                            _game_result = "틀렸습니다!"
                            color = (255, 0, 0)
                    else:
                        _game_result = "문제가 출제되지 않았습니다."
                    
                    last_prediction_time = current_time  # 예측 시간 업데이트

        # 랜드마크 그리기
        if result.left_hand_landmarks:
            mp_drawing.draw_landmarks(img, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if result.right_hand_landmarks:
            mp_drawing.draw_landmarks(img, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if result.pose_landmarks:
            mp_drawing.draw_landmarks(img, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        # 프레임을 웹에 전달
        _, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    if cap is not None and cap.isOpened():
        cap.release()