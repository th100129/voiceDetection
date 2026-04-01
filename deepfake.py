import firebase_admin
from firebase_admin import credentials, storage
import librosa
import tensorflow as tf
import numpy as np

# Firebase Admin SDK 초기화
cred = credentials.Certificate("path/to/serviceAccountKey.json")  # 서비스 계정 키 파일 경로
firebase_admin.initialize_app(cred, {
    'storageBucket': 'deepfake-38f4a.appspot.com'  # Firebase Storage 버킷 이름
})

# 1. Firebase Storage에서 파일 다운로드
bucket = storage.bucket()
blob = bucket.blob('audio/sample_audio.wav')  # Firebase Storage 경로
blob.download_to_filename('downloaded_audio.wav')
print("파일 다운로드 완료")

# 2. 음성 데이터 전처리
audio, sr = librosa.load('downloaded_audio.wav', sr=16000)
mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

# 3. 딥러닝 모델을 사용한 딥페이크 탐지
model = tf.keras.models.load_model('path/to/your_pretrained_model.h5')  # 학습된 모델 경로
mfccs_padded = tf.keras.preprocessing.sequence.pad_sequences([mfccs.T], maxlen=500, padding='post', dtype='float32')
input_data = mfccs_padded.reshape((1, 13, 500, 1))

prediction = model.predict(input_data)
result = "딥페이크" if prediction > 0.5 else "실제 음성"
print(f"예측 결과: {result}")
