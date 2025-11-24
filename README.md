# People Gender Classification with Deep Learning

TMDB 인물 이미지 데이터셋을 사용하여 얼굴 사진으로부터 성별을 분류하는 딥러닝 모델입니다.

## 프로젝트 개요

- **데이터셋**: TMDB People Image Dataset (Hugging Face)
- **프레임워크**: PyTorch
- **모델**: 간단한 CNN (Convolutional Neural Network)
- **작업**: 분류 (Classification) - 성별 분류 (여성/남성)

## 설치 방법

### 1. 필요한 패키지 설치

```bash
pip install -r requirements.txt
```

### 2. 패키지 목록
- `torch`: PyTorch 딥러닝 프레임워크
- `torchvision`: 이미지 전처리 및 변환
- `datasets`: Hugging Face 데이터셋 라이브러리
- `pandas`: 데이터 분석
- `scikit-learn`: Train/Validation 분할
- `Pillow`: 이미지 처리
- `numpy`: 수치 계산

## 사용 방법

### 모델 학습 실행

```bash
python people_gender.py
```

## 모델 구조

### SimpleCNN 아키텍처
```
Input: 128x128x3 RGB 이미지
├── Conv2D (32 filters) + ReLU + MaxPool → 64x64
├── Conv2D (64 filters) + ReLU + MaxPool → 32x32
├── Conv2D (128 filters) + ReLU + MaxPool → 16x16
├── Flatten
├── FC (256) + ReLU + Dropout(0.3)
├── FC (64) + ReLU
└── FC (2) → 성별 분류 (여성/남성)
```

### 하이퍼파라미터
- **입력 크기**: 128x128 RGB
- **배치 크기**: 자동 조정 (데이터 크기에 따라 4, 8, 16, 32)
- **에포크**: 30
- **학습률**: 0.001
- **옵티마이저**: Adam
- **손실 함수**: Cross Entropy Loss
- **평가 지표**: Accuracy (정확도)

## 실행 결과

스크립트는 다음과 같은 작업을 수행합니다:

1. ✅ TMDB 데이터셋 로드 (기본 1000개 샘플, NUM_SAMPLES로 조정 가능)
2. ✅ 사망한 사람 제외 (deathday가 있는 경우)
3. ✅ 성별이 명확한 데이터만 필터링 (여성/남성만)
4. ✅ 이미지 전처리 (Resize, Normalize)
5. ✅ Train/Validation 분할 (80:20, 성별 비율 유지)
6. ✅ CNN 모델 학습 (30 에포크)
7. ✅ 최고 성능 모델 저장 (`best_gender_model.pth`)
8. ✅ 예측 결과 샘플 출력

## 출력 파일

- `best_gender_model.pth`: 최고 성능의 학습된 모델 가중치

## 데이터 전처리

### 이미지 변환
```python
- Resize: 128x128
- ToTensor: [0, 1] 범위로 정규화
- Normalize: ImageNet 평균/표준편차 사용
```

### 데이터 필터링
- 사망한 사람 제외 (deathday가 None이 아닌 경우)
- 성별이 명확한 데이터만 사용 (gender: 1=여성, 2=남성)
- 성별 비율 유지하며 Train/Validation 분할

## 설정 변수

코드 상단에서 다음 변수들을 조정할 수 있습니다:

- `NUM_SAMPLES`: 사용할 데이터 개수 (기본: 1000, None: 전체 데이터)
- `BATCH_SIZE`: 배치 크기 (기본: None, 자동 조정)
- `RANDOM_SEED`: 재현성을 위한 랜덤 시드 (기본: 42)

## 개선 방향

현재는 빠른 프로토타입이므로, 다음과 같이 개선할 수 있습니다:

1. **더 많은 데이터 사용**: NUM_SAMPLES를 None으로 설정하여 전체 데이터셋 사용
2. **Transfer Learning**: ResNet50, EfficientNet 등 사전학습 모델 활용
3. **Data Augmentation**: 회전, 밝기 조정, 좌우 반전 등
4. **하이퍼파라미터 튜닝**: Learning rate, batch size, 모델 크기 조정
5. **Early Stopping**: Validation loss 기반 조기 종료
6. **Learning Rate Scheduler**: 학습률 동적 조정

## 라이선스

MIT License
