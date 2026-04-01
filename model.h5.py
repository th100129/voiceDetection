import tensorflow as tf
from tensorflow.keras import layers, models
import librosa
import numpy as np
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 음성 데이터 로드 및 전처리
def load_audio(file_path):
    audio, sr = librosa.load(file_path, sr=16000)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    print(f'MFCC shape for {file_path}: {mfccs.shape}')  # MFCC 크기 출력
    return mfccs

# CNN 모델 구축
def create_model():
    model = models.Sequential()
    model.add(layers.Input(shape=(13, 500, 1)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  # 딥페이크/실제 여부 이진 분류

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 데이터 로드 및 모델 훈련
audio_files = [
    "C:/Users/user1/Downloads/real.wav",  # 실제 음성
    "C:/Users/user1/Downloads/fake.wav",  # 딥페이크 음성
]

# 데이터 전처리 및 준비
X_train = []
y_train = [0, 1, 1, 1, 0, 0]  # 0: 실제, 1: 딥페이크

for file_path in audio_files:
    audio = load_audio(file_path)
    audio_padded = pad_sequences([audio.T], maxlen=500, padding='post', dtype='float32')
    X_train.append(audio_padded.reshape((13, 500, 1)))

X_train = np.array(X_train)
y_train = np.array(y_train)

# 모델 생성 및 훈련
model = create_model()
model.fit(X_train, y_train, epochs=10)

# 훈련된 모델 저장
model.save('deepfake_detection_model.h5')

# 탐지 함수
def detect_deepfake(audio_path):
    audio = load_audio(audio_path)
    audio = pad_sequences([audio.T], maxlen=500, padding='post', dtype='float32')
    audio = audio.reshape((1, 13, 500, 1))
    prediction = model.predict(audio)
    return "딥페이크" if prediction > 0.5 else "실제 음성"

# 탐지 실행
result = detect_deepfake("C:/Users/user1/Downloads/real.wav") #탐지하기 원하는 파일 선택
print(result)  # 결과 출력
