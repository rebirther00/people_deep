# 상세 Commit 메시지

## 주요 변경사항

### 1. 학습 과제 변경: 나이 예측 → 성별 분류
- **문제점**: 나이 예측은 이미지 촬영 시점과 나이 계산 시점(2025년) 불일치로 학습 의미 없음
- **해결**: 성별 분류로 변경하여 유의미한 학습 과제로 전환
- **변경 파일**: `people_birth.py` → `people_gender.py` (새 파일 생성)

### 2. 모델 아키텍처 개선: SimpleCNN → ResNet18 Transfer Learning
- **이전**: SimpleCNN (128x128 입력, 작은 모델)
- **변경**: ResNet18 Transfer Learning (224x224 입력, ImageNet 사전학습 가중치 사용)
- **효과**: 더 높은 성능과 일반화 능력 확보

### 3. 데이터 품질 개선
- **사망한 사람 제외**: `deathday` 컬럼 체크하여 사망한 사람 데이터 제외
- **데이터 셔플**: Train/Validation 분할 전에 데이터 셔플 수행
- **성별 필터링**: 성별이 명확한 데이터만 사용 (gender: 1=여성, 2=남성)
- **성별 비율 유지**: Train/Validation 분할 시 성별 비율 유지 (stratify 적용)

### 4. 학습 기능 개선
- **Data Augmentation**: Train 데이터에 Flip, Rotation, ColorJitter 적용
- **클래스 가중치**: 불균형 데이터 보정을 위한 클래스 가중치 적용
- **Early Stopping**: Validation loss 개선 없을 시 조기 종료
- **메모리 관리**: 메모리 체크 및 자동 샘플 수 조정
- **GPU 메모리 최적화**: GPU 메모리에 따른 배치 사이즈 자동 조정
- **실행 시간 측정**: 각 단계별 실행 시간 측정 및 출력

### 5. 평가 시스템 구축
- **evaluate_model.py**: 학습된 모델로 데이터셋 평가
  - 학습 데이터 제외 (`training_indices.json` 사용)
  - 랜덤 샘플 평가
  - 클래스별 정확도 계산
  - 결과 이미지 그리드 생성 (`evaluation_results.png`)
  - 오류 분석 (여성→남성, 남성→여성)

- **evaluate_real_images.py**: 로컬 이미지 파일 평가
  - `real_data/` 폴더의 실제 사진 평가
  - 성별 예측 및 신뢰도 출력
  - 결과 이미지 그리드 생성 (`real_data_evaluation_results.png`)
  - 이미지 결과는 영어로 표시 (Female/Male)

### 6. 프로젝트 구조 정리
- **people_birth.py 삭제**: 잘못 구현된 나이 예측 코드 삭제
- **best_age_model.pth 삭제**: 사용하지 않는 모델 파일 삭제
- **PROJECT_STRUCTURE.md 생성**: 전체 프로젝트 구조 문서화
- **README.md 업데이트**: ResNet18 모델 정보로 업데이트

## 파일 변경 내역

### 추가된 파일
- `people_gender.py`: ResNet18 기반 성별 분류 학습 스크립트
- `evaluate_model.py`: 데이터셋 평가 스크립트
- `evaluate_real_images.py`: 로컬 이미지 평가 스크립트
- `PROJECT_STRUCTURE.md`: 프로젝트 구조 문서
- `training_indices.json`: 학습에 사용한 데이터 인덱스 (자동 생성)

### 삭제된 파일
- `people_birth.py`: 잘못 구현된 나이 예측 코드
- `best_age_model.pth`: 사용하지 않는 모델 파일

### 수정된 파일
- `README.md`: 모델 구조 및 사용 방법 업데이트

## 기술적 세부사항

### 모델 구조
- **Backbone**: ResNet18 (ImageNet 사전학습)
- **입력 크기**: 224x224 RGB
- **출력**: 2개 클래스 (여성/남성)
- **FC Layer**: Dropout(0.5) → Linear(512) → ReLU → Dropout(0.3) → Linear(2)

### 전처리 파이프라인
- **Train**: Resize(224x224) + Augmentation + Normalize
- **Validation/Test**: Resize(224x224) + Normalize

### 학습 설정
- **손실 함수**: Cross Entropy Loss (클래스 가중치 적용)
- **옵티마이저**: Adam (lr=0.001)
- **배치 크기**: 자동 조정 (데이터 크기 및 GPU 메모리 고려)
- **에포크**: 30 (Early Stopping 적용, patience=5)

### 평가 기능
- **데이터셋 평가**: 학습 데이터 제외, 랜덤 샘플링
- **로컬 이미지 평가**: 실제 사진 파일 평가
- **결과 시각화**: 그리드 형태의 결과 이미지 생성

## 개선 효과

1. **학습 의미 확보**: 나이 예측의 시점 불일치 문제 해결, 성별 분류로 유의미한 학습
2. **모델 성능 향상**: ResNet18 Transfer Learning으로 더 높은 성능
3. **데이터 품질 향상**: 사망한 사람 제외, 이상치 제거, 셔플 적용
4. **평가 시스템 구축**: 데이터셋 평가 및 로컬 이미지 평가 기능 추가
5. **코드 정리**: 사용하지 않는 파일 삭제, 프로젝트 구조 문서화

## 사용 방법

### 학습
```bash
python3 people_gender.py
# 또는 CPU 강제 실행
python3 people_gender.py --cpu
```

### 평가
```bash
# 데이터셋 평가
python3 evaluate_model.py

# 로컬 이미지 평가
python3 evaluate_real_images.py
```

## 참고사항

- 모든 평가 스크립트는 `people_gender.py`로 학습한 모델(`best_gender_model.pth`)을 사용
- `evaluate_model.py`는 `training_indices.json`을 사용하여 학습 데이터를 제외
- 로컬 이미지 평가 결과는 영어로 표시됨 (Female/Male)

