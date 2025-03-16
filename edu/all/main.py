from flask import Flask, render_template, Response, jsonify
import random
from game1 import generate_frames as generate_frames_game1, actions as actions_game1, set_game_state as set_game1_state, get_game_state as get_game1_state
from game2 import generate_frames as generate_frames_game2, actions as actions_game2, set_game_state as set_game2_state, get_game_state as get_game2_state
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})


# 메인 라우트
@app.route('/')
def index():
    return render_template('game1.html')  # 기본 페이지로 game1.html을 사용

# 게임 1 라우트
@app.route('/game1')
def game1_index():
    return render_template('game1.html')

@app.route('/game1/get_question', methods=['GET'])
def game1_get_question():
    question = random.choice(actions_game1)
    # game1 모듈에 상태 전달
    set_game1_state(question, None)
    return jsonify({"question": question})

@app.route('/game1/get_game_info', methods=['GET'])
def game1_get_game_info():
    # game1 모듈에서 상태 가져오기
    question, result = get_game1_state()
    return jsonify({"question": question, "game_result": result})

@app.route('/game1/video_feed')
def game1_video_feed():
    return Response(generate_frames_game1(),
                    mimetype='multipart/x-mixed-replace; boundary=frame',
                    headers={
                        'Access-Control-Allow-Origin': '*',
                        'Access-Control-Allow-Methods': 'GET, POST',
                        'Access-Control-Allow-Headers': 'Content-Type, Authorization'
                    })

# 게임 2 라우트
@app.route('/game2')
def game2_index():
    return render_template('game2.html')

@app.route('/game2/video_feed')
def game2_video_feed():
    return Response(generate_frames_game2(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame',
                    headers={
                        'Access-Control-Allow-Origin': '*',
                        'Access-Control-Allow-Methods': 'GET, POST',
                        'Access-Control-Allow-Headers': 'Content-Type, Authorization'
                    })

@app.route('/game2/get_question', methods=['GET'])
def game2_get_question():
    question = random.choice(actions_game2)
    # game2 모듈에 상태 전달
    set_game2_state(question, None)
    return jsonify({"question": question})

@app.route('/game2/get_game_info', methods=['GET'])
def game2_get_game_info():
    # game2 모듈에서 상태 가져오기
    question, result = get_game2_state()
    return jsonify({"question": question, "game_result": result})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)