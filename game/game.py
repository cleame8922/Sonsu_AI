from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import modules.holistic_module as hm
from modules.utils import Vector_Normalization
from PIL import ImageFont, ImageDraw, Image
import random
import pymysql  # MySQL 연동

# Flask 앱 초기화
app = Flask(__name__)

# MySQL 연결
db = pymysql.connect(
    host="localhost",      # 데이터베이스 서버 주소
    user="junsseok",       # MySQL 사용자
    password="1234",       # MySQL 비밀번호
    database="sonsu",      # 데이터베이스 이름
    charset="utf8mb4",
    cursorclass=pymysql.cursors.DictCursor
)

# 모델 및 MediaPipe 초기화
actions = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
           'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ',
           'ㅐ', 'ㅒ', 'ㅔ', 'ㅖ', 'ㅢ', 'ㅚ', 'ㅟ']
seq_length = 10
detector = hm.HolisticDetector(min_detection_confidence=0.3)
interpreter = tf.lite.Interpreter(model_path="models/multi_hand_gesture_classifier.tflite")
interpreter.allocate_tensors()

# 입력 및 출력 정보
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

seq = []
action_seq = []
last_action = None
current_question = None
game_result = None

# 정답 및 오답 저장 함수
def save_answer(word, is_correct):
    try:
        with db.cursor() as cursor:
            sql = "INSERT INTO correct_answers (word, ox) VALUES (%s, %s)"
            cursor.execute(sql, (word, int(is_correct)))  # Boolean 대신 1/0 저장
        db.commit()
    except Exception as e:
        print("DB 저장 오류:", e)

# 비디오 스트리밍 처리
def process_frame():
    global seq, action_seq, last_action, game_result
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break

        img = cv2.flip(img, 1)  # 좌우 반전
        img = detector.findHolistic(img, draw=True)
        _, right_hand_lmList = detector.findLefthandLandmark(img)

        if right_hand_lmList is not None:
            joint = np.zeros((42, 2))
            for j, lm in enumerate(right_hand_lmList.landmark):
                joint[j] = [lm.x, lm.y]
            vector, angle_label = Vector_Normalization(joint)
            d = np.concatenate([vector.flatten(), angle_label.flatten()])
            seq.append(d)

            if len(seq) < seq_length:
                continue

            input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            y_pred = interpreter.get_tensor(output_details[0]['index'])
            i_pred = int(np.argmax(y_pred))
            conf = y_pred[0][i_pred]

            if conf > 0.9 and 0 <= i_pred < len(actions):
                action = actions[i_pred]
                action_seq.append(action)

                if len(action_seq) >= 3 and action_seq[-1] == action_seq[-2] == action_seq[-3]:
                    this_action = action
                    if last_action != this_action:
                        last_action = this_action

                        if current_question is not None:
                            if this_action == current_question:
                                game_result = "정답입니다!"
                                save_answer(this_action, True)  # 정답 저장
                            else:
                                game_result = "틀렸습니다!"
                                save_answer(this_action, False)  # 오답 저장

        _, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('game.html')

@app.route('/video_feed')
def video_feed():
    return Response(process_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_question', methods=['GET'])
def get_question():
    global current_question, game_result
    current_question = random.choice(actions)
    game_result = None
    return jsonify({"question": current_question})

if __name__ == '__main__':
    app.run(debug=True)
