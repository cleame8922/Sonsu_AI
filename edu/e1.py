import cv2
import mediapipe as mp
import numpy as np
import time, os

actions = ['어머니',
    '아버지',
    '동생']  # 학습할 수어 동작
seq_length = 30  # 시퀀스 길이
secs_for_action = 30  # 각 동작을 30초 동안 촬영

# MediaPipe Holistic 모델 로드
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

created_time = int(time.time())
os.makedirs('dataset', exist_ok=True)

while cap.isOpened():
    for idx, action in enumerate(actions):
        data = []

        ret, img = cap.read()
        img = cv2.flip(img, 1)  # 좌우 반전
        
        cv2.putText(img, f'Waiting for collecting {action.upper()} action...', org=(10, 30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
        cv2.imshow('img', img)
        cv2.waitKey(3000)

        start_time = time.time()

        while time.time() - start_time < secs_for_action:
            ret, img = cap.read()
            img = cv2.flip(img, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = holistic.process(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            joint_list = []

            # 왼손 랜드마크 저장 (21개)
            if result.left_hand_landmarks:
                for lm in result.left_hand_landmarks.landmark:
                    joint_list.append([lm.x, lm.y, lm.z])
            else:
                joint_list.extend([[0, 0, 0]] * 21)  # 손이 없을 경우 0으로 채움

            # 오른손 랜드마크 저장 (21개)
            if result.right_hand_landmarks:
                for lm in result.right_hand_landmarks.landmark:
                    joint_list.append([lm.x, lm.y, lm.z])
            else:
                joint_list.extend([[0, 0, 0]] * 21)

            # 몸(상체) 랜드마크 저장 (33개)
            if result.pose_landmarks:
                for lm in result.pose_landmarks.landmark:
                    joint_list.append([lm.x, lm.y, lm.z])
            else:
                joint_list.extend([[0, 0, 0]] * 33)

            if joint_list:
                joint_list = np.array(joint_list).flatten()
                joint_list = np.append(joint_list, idx)  # Label 추가
                data.append(joint_list)

            # 랜드마크 시각화
            mp_drawing.draw_landmarks(img, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(img, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(img, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

            cv2.imshow('img', img)
            if cv2.waitKey(1) == ord('q'):
                break

        data = np.array(data)
        print(action, data.shape)
        np.save(os.path.join('dataset', f'raw_{action}'), data)

        full_seq_data = []
        for seq in range(len(data) - seq_length):
            full_seq_data.append(data[seq:seq + seq_length])

        full_seq_data = np.array(full_seq_data)
        print(action, full_seq_data.shape)
        np.save(os.path.join('dataset', f'seq_{action}'), full_seq_data)

    break

cap.release()
cv2.destroyAllWindows()
