from flask import Flask, request, Response
import librosa
import numpy as np
import tensorflow as tf
import logging

app = Flask(__name__)

# 로깅 설정
logging.basicConfig(level=logging.DEBUG)

# 모델 파일 경로 설정
MODEL_PATH = "C:/Users/user1/PycharmProjects/VoiceDetection/model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

@app.route('/analyze', methods=['POST'])
def analyze_audio():
    try:
        # HTTP POST 요청에서 음성 파일 받기
        file = request.files['file']
        audio, sr = librosa.load(file, sr=16000)  # 파일을 librosa로 로드
        logging.info(f"Loaded audio with sample rate {sr} and shape {audio.shape}")

        # MFCC 특성 추출
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        logging.info(f"Extracted MFCCs with shape {mfccs.shape}")

        # 데이터를 모델 입력 형식에 맞게 조정
        if mfccs.shape[1] < 500:
            mfccs = np.pad(mfccs, ((0, 0), (0, 500 - mfccs.shape[1])), mode='constant')
        else:
            mfccs = mfccs[:, :500]  # 너무 긴 경우 자르기

        audio_input = mfccs.T[:500].reshape((1, 500, 13, 1))

        # 딥페이크 여부 예측
        prediction = model.predict(audio_input)
        result = "딥페이크" if prediction > 0.5 else "실제 음성"
        return result

    except Exception as e:
        logging.error(f"Error processing audio: {e}")
        return str(e)

if __name__ == '__main__':
    app.run(port=5000)
