"""
학습된 모델을 사용하여 랜덤 샘플 평가 스크립트
"""
from datasets import load_dataset
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
import random

# ================================================================================
# 설정 변수
# ================================================================================
MODEL_PATH = 'best_gender_model.pth'  # 평가할 모델 파일 경로
NUM_TEST_SAMPLES = 100  # 평가할 랜덤 샘플 개수 (원하는 숫자로 변경 가능: 100, 500, 1000 등)
RANDOM_SEED = None  # None: 매번 랜덤, 숫자: 재현성을 위한 고정 시드
BATCH_SIZE = 32  # 배치 크기

# ================================================================================
# 1. 데이터 로드 및 전처리
# ================================================================================
print("=" * 80)
print("모델 평가: 랜덤 샘플 평가")
print("=" * 80)

# 랜덤 시드 설정 (None이면 매번 다른 랜덤 샘플 선택)
if RANDOM_SEED is None:
    # 현재 시간 기반으로 랜덤 시드 생성
    import time
    RANDOM_SEED = int(time.time() * 1000000) % (2**32)
    print(f"랜덤 시드 자동 생성: {RANDOM_SEED}")
else:
    print(f"고정 랜덤 시드 사용: {RANDOM_SEED}")

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

print(f"\n모델 파일: {MODEL_PATH}")
print(f"평가 샘플 수: {NUM_TEST_SAMPLES}개")
print(f"랜덤 시드: {RANDOM_SEED}")

# 데이터셋 로드
print("\n데이터셋 로드 중...")
ds = load_dataset("ashraq/tmdb-people-image")
total_size = len(ds['train'])
print(f"전체 데이터셋 크기: {total_size:,}개")

# 메모리 효율성을 위해 충분한 샘플을 먼저 선택
# 필터링 후에도 NUM_TEST_SAMPLES만큼 남도록 여유있게 샘플링
# (사망한 사람, 성별 미지정 등을 고려하여 3-4배 정도 샘플링)
SAMPLE_MULTIPLIER = 4  # 필터링 후 손실을 고려한 배수
sample_size = max(NUM_TEST_SAMPLES * SAMPLE_MULTIPLIER, 1000)  # 최소 1000개

if sample_size < total_size:
    print(f"메모리 효율성을 위해 먼저 {sample_size}개 샘플링 중...")
    # 랜덤 샘플링을 위한 인덱스 생성
    indices = list(range(total_size))
    random.shuffle(indices)
    selected_indices = indices[:sample_size]
    selected_indices.sort()  # 정렬하여 순차 접근으로 성능 향상
    # 선택된 인덱스만 로드하여 DataFrame 생성
    df = pd.DataFrame(ds['train'].select(selected_indices))
    print(f"샘플링된 데이터 개수: {len(df)}개")
else:
    print("전체 데이터셋을 DataFrame으로 변환 중... (시간이 걸릴 수 있습니다)")
    df = pd.DataFrame(ds['train'])
    print(f"전체 데이터 개수: {len(df)}개")

# 사망한 사람 제외
if 'deathday' in df.columns:
    before_death_filter = len(df)
    df = df[df['deathday'].isna()].copy()
    after_death_filter = len(df)
    print(f"사망한 사람 제외: {before_death_filter - after_death_filter}명 제외됨")

# 성별이 명확한 데이터만 사용 (1: 여성, 2: 남성)
df_filtered = df[df['gender'].isin([1, 2])].copy()
print(f"성별이 명확한 데이터: {len(df_filtered)}개")

if len(df_filtered) == 0:
    raise ValueError("평가에 사용할 수 있는 데이터가 없습니다!")

# 최종 랜덤 샘플링
if NUM_TEST_SAMPLES > len(df_filtered):
    print(f"경고: 요청한 샘플 수({NUM_TEST_SAMPLES})가 사용 가능한 데이터({len(df_filtered)})보다 많습니다.")
    print(f"전체 데이터({len(df_filtered)}개)를 사용합니다.")
    test_df = df_filtered.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
else:
    test_df = df_filtered.sample(n=NUM_TEST_SAMPLES, random_state=RANDOM_SEED).reset_index(drop=True)
    print(f"최종 랜덤 샘플링 완료: {len(test_df)}개 선택")

# 성별 분포 확인
female_count = len(test_df[test_df['gender'] == 1])
male_count = len(test_df[test_df['gender'] == 2])
print(f"  - 여성: {female_count}명, 남성: {male_count}명")


# ================================================================================
# 2. Dataset 클래스 정의
# ================================================================================
class PeopleGenderDataset(Dataset):
    """TMDB 인물 이미지와 성별을 위한 Dataset 클래스"""
    
    def __init__(self, data, transform=None):
        self.data = data.reset_index(drop=True)
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 이미지 로드
        image = self.data.iloc[idx]['image']
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 성별 정보 (1: 여성 -> 0, 2: 남성 -> 1)
        gender_id = self.data.iloc[idx]['gender']
        gender_label = 0 if gender_id == 1 else 1  # 여성: 0, 남성: 1
        gender_label = torch.tensor(gender_label, dtype=torch.long)
        
        # 이미지 전처리 적용
        if self.transform:
            image = self.transform(image)
        
        return image, gender_label


# 이미지 전처리 (평가용 - Augmentation 없음)
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet은 224x224 입력 사용
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Dataset 및 DataLoader 생성
test_dataset = PeopleGenderDataset(test_df, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"\n테스트 배치 개수: {len(test_loader)}")


# ================================================================================
# 3. 모델 정의 및 로드
# ================================================================================
print("\n" + "=" * 80)
print("모델 로드")
print("=" * 80)

def create_resnet_model(num_classes=2, pretrained=False):
    """ResNet18 기반 모델 생성"""
    model = models.resnet18(pretrained=pretrained)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    return model

# 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용 디바이스: {device}")

# 모델 생성 및 가중치 로드
model = create_resnet_model(num_classes=2, pretrained=False).to(device)
print(f"모델 파일 로드 중: {MODEL_PATH}")
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print("모델 로드 완료!")


# ================================================================================
# 4. 평가 실행
# ================================================================================
print("\n" + "=" * 80)
print("평가 실행")
print("=" * 80)

gender_names = {0: "여성", 1: "남성"}
correct = 0
total = 0
class_correct = [0, 0]  # [여성, 남성]
class_total = [0, 0]
all_predictions = []
all_labels = []

print("\n예측 결과:")
print("-" * 80)

with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
        
        # 배치 내 각 샘플에 대해 결과 출력
        for i in range(len(labels)):
            label = labels[i].item()
            pred = predictions[i].item()
            confidence = torch.softmax(outputs[i], dim=0)[pred].item() * 100
            
            actual_gender = gender_names[label]
            predicted_gender = gender_names[pred]
            is_correct = (label == pred)
            
            if is_correct:
                correct += 1
                class_correct[label] += 1
                symbol = "✓"
            else:
                symbol = "✗"
            
            total += 1
            class_total[label] += 1
            all_predictions.append(pred)
            all_labels.append(label)
            
            # 결과 출력 (모든 샘플)
            print(f"{symbol} 실제: {actual_gender:3s} | 예측: {predicted_gender:3s} | "
                  f"신뢰도: {confidence:.1f}%")

# 전체 통계
print("\n" + "=" * 80)
print("평가 결과 요약")
print("=" * 80)

overall_accuracy = 100.0 * correct / total
female_accuracy = 100.0 * class_correct[0] / class_total[0] if class_total[0] > 0 else 0.0
male_accuracy = 100.0 * class_correct[1] / class_total[1] if class_total[1] > 0 else 0.0

print(f"\n전체 정확도: {overall_accuracy:.2f}% ({correct}/{total})")
print(f"  - 여성 정확도: {female_accuracy:.2f}% ({class_correct[0]}/{class_total[0]})")
print(f"  - 남성 정확도: {male_accuracy:.2f}% ({class_correct[1]}/{class_total[1]})")

# 오류 분석
errors = total - correct
print(f"\n오류 개수: {errors}개")
if errors > 0:
    print("\n오류 분석:")
    female_to_male = sum(1 for i in range(len(all_labels)) 
                        if all_labels[i] == 0 and all_predictions[i] == 1)
    male_to_female = sum(1 for i in range(len(all_labels)) 
                        if all_labels[i] == 1 and all_predictions[i] == 0)
    print(f"  - 여성 → 남성 오류: {female_to_male}개")
    print(f"  - 남성 → 여성 오류: {male_to_female}개")

print("\n" + "=" * 80)
print("평가 완료!")
print("=" * 80)


