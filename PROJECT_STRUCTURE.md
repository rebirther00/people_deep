# 프로젝트 구조 문서

## 전체 개요

이 프로젝트는 TMDB 인물 이미지 데이터셋을 사용하여 얼굴 사진으로부터 성별을 분류하는 딥러닝 모델입니다.

## 파일 구조 및 역할

### 1. 학습 파일

#### `people_gender.py` ✅ **현재 사용 중**
- **역할**: 성별 분류 모델 학습
- **모델**: ResNet18 (Transfer Learning, ImageNet 사전학습 가중치 사용)
- **입력 크기**: 224x224
- **출력**: `best_gender_model.pth` (학습된 모델 가중치)
- **부가 출력**: `training_indices.json` (학습에 사용한 데이터 인덱스)
- **주요 기능**:
  - 메모리 체크 및 자동 샘플 수 조정
  - GPU 메모리 고려 배치 사이즈 자동 조정
  - Data Augmentation (Train: Flip, Rotation, ColorJitter)
  - 클래스 가중치 적용 (불균형 데이터 보정)
  - Early Stopping
  - 단계별 실행 시간 측정

### 2. 평가 파일

#### `evaluate_model.py` ✅ **현재 사용 중**
- **역할**: 학습된 모델로 데이터셋 평가
- **모델**: ResNet18 (학습된 가중치 로드)
- **입력 크기**: 224x224
- **모델 파일**: `best_gender_model.pth`
- **주요 기능**:
  - 학습 데이터 제외 (`training_indices.json` 사용)
  - 랜덤 샘플 평가
  - 클래스별 정확도 계산
  - 결과 이미지 그리드 생성 (`evaluation_results.png`)
  - 오류 분석 (여성→남성, 남성→여성)

#### `evaluate_real_images.py` ✅ **현재 사용 중**
- **역할**: 로컬 이미지 파일 평가
- **모델**: ResNet18 (학습된 가중치 로드)
- **입력 크기**: 224x224
- **모델 파일**: `best_gender_model.pth`
- **입력**: `real_data/` 폴더의 이미지 파일들
- **주요 기능**:
  - 로컬 이미지 파일 자동 검색 (jpg, jpeg, png)
  - 성별 예측 및 신뢰도 출력
  - 결과 이미지 그리드 생성 (`real_data_evaluation_results.png`)

### 3. 삭제된 파일

#### `people_birth.py` ❌ **삭제됨**
- **이유**: 잘못 구현된 나이 예측 코드 (이미지 촬영 시점과 나이 계산 시점 불일치 문제)
- **상태**: 성별 분류로 전환하면서 삭제됨

#### `best_age_model.pth` ❌ **삭제됨**
- **이유**: people_birth.py로 생성된 모델 파일 (더 이상 사용하지 않음)

#### `check_memory.py`
- **역할**: 메모리 체크 유틸리티 (참고용)

## 데이터 흐름

```
1. 데이터셋 로드 (ashraq/tmdb-people-image)
   ↓
2. 전처리
   - 사망한 사람 제외 (deathday 체크)
   - 성별이 명확한 데이터만 사용 (gender: 1=여성, 2=남성)
   - 데이터 셔플
   ↓
3. Train/Validation 분할 (80:20, 성별 비율 유지)
   ↓
4. 학습 (people_gender.py)
   - ResNet18 Transfer Learning
   - Data Augmentation (Train만)
   - 클래스 가중치 적용
   ↓
5. 모델 저장 (best_gender_model.pth)
   학습 인덱스 저장 (training_indices.json)
   ↓
6. 평가
   - evaluate_model.py: 데이터셋 평가 (학습 데이터 제외)
   - evaluate_real_images.py: 로컬 이미지 평가
```

## 모델 구조

### ResNet18 기반 모델
```
Input: 224x224x3 RGB 이미지
  ↓
ResNet18 Backbone (ImageNet 사전학습)
  ↓
FC Layer 구조:
  - Dropout(0.5)
  - Linear(512, 512)
  - ReLU
  - Dropout(0.3)
  - Linear(512, 2)
  ↓
Output: [여성 확률, 남성 확률]
```

## 전처리 파이프라인

### 학습 시 (Train)
```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### 평가 시 (Validation/Test)
```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

## 설정 변수

### people_gender.py
- `NUM_SAMPLES`: 학습 데이터 개수 (기본값: 20000)
- `BATCH_SIZE`: 배치 크기 (None: 자동 조정)
- `RANDOM_SEED`: 재현성 시드 (42)
- `MEMORY_SAFETY_MARGIN`: 메모리 사용 한계 (0.5 = 50%)

### evaluate_model.py
- `NUM_TEST_SAMPLES`: 평가 샘플 수 (기본값: 50)
- `RANDOM_SEED`: None (매번 랜덤) 또는 고정 값

### evaluate_real_images.py
- `IMAGE_DIR`: 평가할 이미지 폴더 (기본값: 'real_data')
- `BATCH_SIZE`: 배치 크기 (기본값: 8)

## 출력 파일

### 학습 결과
- `best_gender_model.pth`: 최고 성능 모델 가중치
- `training_indices.json`: 학습에 사용한 데이터 인덱스

### 평가 결과
- `evaluation_results.png`: 데이터셋 평가 결과 이미지 (evaluate_model.py)
- `real_data_evaluation_results.png`: 로컬 이미지 평가 결과 이미지 (evaluate_real_images.py)

## 사용 방법

### 1. 모델 학습
```bash
python3 people_gender.py
# 또는 CPU 강제 실행
python3 people_gender.py --cpu
```

### 2. 데이터셋 평가
```bash
python3 evaluate_model.py
```

### 3. 로컬 이미지 평가
```bash
# real_data 폴더에 이미지 파일을 넣고 실행
python3 evaluate_real_images.py
```

## 주의사항

1. **모델 호환성**: 모든 평가 스크립트는 `people_gender.py`로 학습한 모델(`best_gender_model.pth`)을 사용해야 합니다.
2. **입력 크기**: 모든 스크립트는 224x224 입력을 사용합니다 (ResNet18 표준).
3. **데이터 분리**: `evaluate_model.py`는 `training_indices.json`을 사용하여 학습 데이터를 제외합니다.
4. **메모리 관리**: `people_gender.py`는 메모리를 자동으로 체크하고 샘플 수를 조정합니다.

## 삭제된 파일

- `people_birth.py`: 잘못 구현된 나이 예측 코드로 인해 삭제됨
- `best_age_model.pth`: people_birth.py로 생성된 모델 파일로 삭제됨

## 향후 개선 사항

1. 파일명 통일: `people_birth.py` 삭제 또는 이름 변경
2. 설정 파일 분리: 하이퍼파라미터를 별도 config 파일로 관리
3. 로깅 시스템: 학습 과정을 더 체계적으로 기록

