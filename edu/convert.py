import tensorflow as tf

# 모델 로드
loaded_model = tf.keras.models.load_model("models/model1.h5")  # 모델 경로 확인

# TFLite 변환기 생성
converter = tf.lite.TFLiteConverter.from_keras_model(loaded_model)

# TensorFlow Lite 기본 연산 + TensorFlow 연산을 허용
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,   # 기본 연산
    tf.lite.OpsSet.SELECT_TF_OPS      # 추가 TensorFlow 연산 허용
]

# 변환 실행
tflite_model = converter.convert()

# 변환된 모델 저장
with open("model1.tflite", "wb") as f:
    f.write(tflite_model)

print("변환 완료: model.tflite 저장됨.")
