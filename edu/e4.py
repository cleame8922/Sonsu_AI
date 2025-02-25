import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from PIL import ImageFont, ImageDraw, Image

# 모델 및 데이터 정보
model_path = 'model.h5'  # 저장한 h5 파일 경로
actions = ['안녕하세요', '사랑합니다', '감사합니다']  # 학습한 동작 리스트
seq_length = 30  # 모델 학습 시 사용한 시퀀스 길이

# 한글 폰트
font_path = "malgun.ttf"  # 또는 "NanumGothic.ttf"
font = ImageFont.truetype(font_path, 30)

# 모델 로드
model = tf.keras.models.load_model(model_path)

# MediaPipe 모델 로드
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
seq = []  # 시퀀스 데이터 저장 리스트

def draw_text(img, text, position, font, color=(0, 255, 0)):
    """PIL을 이용해 한글을 출력하는 함수"""
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=color)
    return np.array(img_pil)

while cap.isOpened():
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

    if joint_list:
        joint_list = np.array(joint_list).flatten()
        seq.append(joint_list)

        # 시퀀스 길이 유지
        if len(seq) > seq_length:
            seq.pop(0)

        # 예측 수행
        if len(seq) == seq_length:
            input_data = np.expand_dims(np.array(seq), axis=0)
            prediction = model.predict(input_data)[0]
            predicted_action = actions[np.argmax(prediction)]
            confidence = np.max(prediction)

            # 한글 예측 결과를 이미지에 표시
            img = draw_text(img, f'{predicted_action} ({confidence:.2f})', (10, 50), font)

    # 랜드마크 그리기
    mp_drawing.draw_landmarks(img, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(img, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(img, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

    cv2.imshow('Sign Language Recognition', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
