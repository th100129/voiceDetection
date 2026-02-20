import firebase_admin
import tensorflow as tf
from firebase_admin import credentials, storage
from fastapi import FastAPI, HTTPException
import uvicorn
import librosa
import numpy as np

# Firebase Admin SDK 초기화
cred = credentials.Certificate("C:/Users/user1/Downloads/deepfake-38f4a-firebase-adminsdk-6lmmb-4675a7110a.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'deepfake-38f4a.appspot.com'  # Firebase Storage 버킷 이름
})

# 딥페이크 탐지 모델 로드
model = tf.keras.models.load_model('C:/Users/user1/PycharmProjects/VoiceDetection/model.h5')

# FastAPI 앱 초기화
app = FastAPI()

# Firebase Storage에서 파일 다운로드 함수
def download_audio_from_firebase(file_name, local_path='downloaded_audio.wav'):
    bucket = storage.bucket()
    blob = bucket.blob(file_name)
    blob.download_to_filename(local_path)
    return local_path

# 음성 데이터 전처리 함수
def preprocess_audio(file_path):
    audio, sr = librosa.load(file_path, sr=16000)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfccs_padded = tf.keras.preprocessing.sequence.pad_sequences([mfccs.T], maxlen=500, padding='post', dtype='float32')
    input_data = mfccs_padded.reshape((1, 13, 500, 1))
    return input_data

# 딥페이크 탐지 API 엔드포인트
@app.post("/analyze-audio/")
async def analyze_audio(file_name: str):
    try:
        # Firebase에서 음성 파일 다운로드
        local_file_path = download_audio_from_firebase(f"audio/{file_name}")

        # 음성 파일 전처리
        input_data = preprocess_audio(local_file_path)

        # 딥페이크 판별
        prediction = model.predict(input_data)
        result = "딥페이크" if prediction > 0.5 else "실제 음성"

        return {"result": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 서버 실행
if __name__ == "__main__":
    uvicorn.run(app, host="192.168.52.1", port=8000)
