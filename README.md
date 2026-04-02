# Voice Deepfake Detection System

## Overview
딥페이크 음성을 탐지하기 위한 AI 기반 음성 분석 시스템입니다.  
MFCC(Mel-Frequency Cepstral Coefficients) 특징 추출과 CNN 모델을 활용하여  
실제 음성과 딥페이크 음성을 구분합니다.

---

## Key Features
- 음성 데이터 전처리 (librosa 기반)
- MFCC 특징 추출
- CNN 기반 분류 모델 구현
- Flask 서버를 통한 모델 서빙
- WAV 변환 및 오디오 처리 파이프라인 구축

---

## Tech Stack
- **Language**: Python  
- **AI/ML**: TensorFlow, Keras  
- **Audio Processing**: librosa  
- **Backend**: Flask  

---

## Architecture
Audio Input → MFCC Feature Extraction → CNN Model → Prediction Output


---

## Project Structure
VoiceDetection/
├── main.py # 모델 학습 및 실행
├── deepfake.py # 딥페이크 탐지 로직
├── flask_app.py # 서버 API
├── server.py # 서버 실행
├── mp4Towav.py # 오디오 변환
├── model.h5.py # 모델 정의
├── .gitignore

---

## How It Works

1. 음성 파일을 입력받음  
2. librosa를 통해 MFCC 특징 추출  
3. CNN 모델에 입력  
4. 딥페이크 여부 예측  

---

## Model Details
- Input: MFCC (13 × N)
- Architecture:
  - Conv2D + ReLU
  - MaxPooling
  - Dense Layer
- Output: Binary Classification (Real / Fake)

---

## Key Contribution
- 음성 데이터를 MFCC로 변환하여 모델 입력 최적화
- CNN 구조를 활용한 딥페이크 탐지 모델 직접 구현
- 오디오 처리부터 모델 서빙까지 End-to-End 파이프라인 구축

---

## Future Improvements
- Transformer 기반 모델 적용
- 데이터셋 확장 및 일반화 성능 개선
- 실시간 음성 스트리밍 탐지 시스템 개발

---

## Note
- 모델 파일(.h5)은 용량 문제로 GitHub에 포함되지 않았습니다.
