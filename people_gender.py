from datasets import load_dataset
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import random
import psutil
import os
import argparse
import time

# ================================================================================
# 명령줄 인자 파싱
# ================================================================================
parser = argparse.ArgumentParser(description='성별 분류 모델 학습')
parser.add_argument('-cpu', '--cpu', action='store_true', 
                    help='CPU로 강제 실행 (기본값: GPU 사용 가능 시 GPU 사용)')
args = parser.parse_args()

# ================================================================================
# 설정 변수
# ================================================================================
NUM_SAMPLES = 20000  # None: 전체 데이터 사용, 숫자: 해당 개수만큼만 사용 (예: 100, 500, 1000)
BATCH_SIZE = None   # None: 자동 조정, 숫자: 고정 배치 사이즈 (예: 8, 16, 32, 64)
RANDOM_SEED = 42    # 재현성을 위한 랜덤 시드
MEMORY_SAFETY_MARGIN = 0.5  # 사용 가능 메모리의 몇 %까지 사용할지 (0.5 = 50%)
SAMPLE_SIZE_PER_GB = 979  # GB당 샘플 수 (체크 결과 기반)

# 전체 실행 시간 측정 시작
total_start_time = time.time()

# ================================================================================
# 1. 데이터 로드 및 전처리
# ================================================================================
print("=" * 80)
print("1단계: 데이터셋 로드 및 전처리")
print("=" * 80)
step1_start_time = time.time()

# 랜덤 시드 설정 (재현성)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available() and not args.cpu:
    torch.cuda.manual_seed_all(RANDOM_SEED)

# 메모리 체크 및 샘플 수 자동 조정
def check_memory_and_adjust_samples(requested_samples, total_size):
    """메모리를 체크하고 안전한 샘플 수를 반환"""
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024**3)  # GB 단위
    safe_memory_gb = available_gb * MEMORY_SAFETY_MARGIN
    
    # 샘플당 메모리 사용량 (GB)
    memory_per_sample_gb = 1.0 / SAMPLE_SIZE_PER_GB
    
    # 안전하게 사용 가능한 샘플 수
    max_safe_samples = int(safe_memory_gb / memory_per_sample_gb)
    
    print(f"\n[메모리 체크]")
    print(f"  사용 가능 메모리: {available_gb:.2f} GB")
    print(f"  안전 사용 한계 ({MEMORY_SAFETY_MARGIN*100:.0f}%): {safe_memory_gb:.2f} GB")
    print(f"  샘플당 메모리: {memory_per_sample_gb*1024:.2f} MB")
    print(f"  최대 안전 샘플 수: {max_safe_samples:,}개")
    
    if requested_samples is None:
        # 전체 데이터 사용 요청 시
        estimated_memory = total_size * memory_per_sample_gb
        if estimated_memory > safe_memory_gb:
            print(f"\n⚠️  경고: 전체 데이터({total_size:,}개) 사용 시 예상 메모리: {estimated_memory:.2f} GB")
            print(f"   안전 한계({safe_memory_gb:.2f} GB)를 초과합니다!")
            print(f"   권장: NUM_SAMPLES를 {max_safe_samples:,}개 이하로 설정하세요.")
            return None, max_safe_samples
        return None, None
    else:
        # 특정 샘플 수 요청 시
        estimated_memory = requested_samples * memory_per_sample_gb
        if estimated_memory > safe_memory_gb:
            print(f"\n⚠️  경고: 요청한 샘플 수({requested_samples:,}개)의 예상 메모리: {estimated_memory:.2f} GB")
            print(f"   안전 한계({safe_memory_gb:.2f} GB)를 초과합니다!")
            print(f"   자동 조정: {max_safe_samples:,}개로 제한합니다.")
            return max_safe_samples, max_safe_samples
        else:
            print(f"  요청 샘플 수: {requested_samples:,}개 (예상 메모리: {estimated_memory:.2f} GB) ✓")
            return requested_samples, None

# 데이터셋 로드
print("데이터셋 로드 중...")
ds = load_dataset("ashraq/tmdb-people-image")
total_size = len(ds['train'])
print(f"전체 데이터셋 크기: {total_size:,}개")

# 메모리 체크 및 샘플 수 조정
adjusted_samples, max_safe = check_memory_and_adjust_samples(NUM_SAMPLES, total_size)
if adjusted_samples is not None and adjusted_samples != NUM_SAMPLES:
    print(f"\n샘플 수 자동 조정: {NUM_SAMPLES:,}개 → {adjusted_samples:,}개")
    NUM_SAMPLES = adjusted_samples

# 샘플링할 인덱스 선택 (메모리 효율성을 위해 먼저 샘플링)
if NUM_SAMPLES is not None and NUM_SAMPLES < total_size:
    # 랜덤 샘플링을 위한 인덱스 생성
    indices = list(range(total_size))
    random.shuffle(indices)
    selected_indices = indices[:NUM_SAMPLES]
    selected_indices.sort()  # 정렬하여 순차 접근으로 성능 향상
    print(f"샘플링 중: {NUM_SAMPLES}개 선택...")
    # 선택된 인덱스만 로드하여 DataFrame 생성
    df = pd.DataFrame(ds['train'].select(selected_indices))
    # 원본 데이터셋 인덱스를 DataFrame에 추가 (평가 시 제외하기 위해)
    df['original_index'] = selected_indices
    print(f"샘플링된 데이터 개수: {len(df)}개")
    # 샘플링된 데이터도 셔플 (랜덤 샘플링이지만 DataFrame 내에서도 셔플)
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
else:
    # 전체 데이터 사용 시에도 배치로 처리하여 메모리 효율성 향상
    print("전체 데이터셋을 DataFrame으로 변환 중... (시간이 걸릴 수 있습니다)")
    df = pd.DataFrame(ds['train'])
    # 원본 인덱스 추가
    df['original_index'] = range(len(df))
    print(f"전체 데이터 개수: {len(df)}개")
    # 데이터 셔플 (분할 전에 셔플 수행)
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

print(f"컬럼 목록: {df.columns.tolist()}")

# 사망한 사람 제외 (deathday가 None이 아닌 경우 제외)
if 'deathday' in df.columns:
    before_death_filter = len(df)
    df = df[df['deathday'].isna()].copy()
    after_death_filter = len(df)
    print(f"\n사망한 사람 제외: {before_death_filter - after_death_filter}명 제외됨")
    print(f"생존자 데이터: {after_death_filter}개")

# 성별 정보 확인 및 필터링
if 'gender' not in df.columns:
    raise ValueError("성별(gender) 컬럼을 찾을 수 없습니다!")

# 성별 분포 확인 (0: 미지정, 1: 여성, 2: 남성, 3: 기타)
print(f"\n성별 분포:")
gender_counts = df['gender'].value_counts().sort_index()
for gender_id, count in gender_counts.items():
    gender_name = {0: "미지정", 1: "여성", 2: "남성", 3: "기타"}.get(gender_id, f"기타({gender_id})")
    print(f"  {gender_name} (ID: {gender_id}): {count}명")

# 성별이 명확한 데이터만 사용 (1: 여성, 2: 남성)
df_filtered = df[df['gender'].isin([1, 2])].copy()
print(f"\n학습에 사용할 데이터 (여성/남성만): {len(df_filtered)}개")

if len(df_filtered) == 0:
    raise ValueError("학습에 사용할 수 있는 데이터가 없습니다!")

step1_time = time.time() - step1_start_time
print(f"\n[1단계 완료] 소요 시간: {step1_time:.2f}초 ({step1_time/60:.2f}분)")

# ================================================================================
# 2. PyTorch Dataset 클래스 정의
# ================================================================================
print("\n" + "=" * 80)
print("2단계: PyTorch Dataset 클래스 정의")
print("=" * 80)
step2_start_time = time.time()


class PeopleGenderDataset(Dataset):
    """TMDB 인물 이미지와 성별을 위한 Dataset 클래스"""
    
    def __init__(self, data, transform=None):
        """
        Args:
            data: DataFrame (image, gender 컬럼 포함)
            transform: 이미지 변환 (torchvision.transforms)
        """
        self.data = data.reset_index(drop=True)
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 이미지 로드 (PIL Image 형식)
        image = self.data.iloc[idx]['image']
        
        # 이미지가 PIL Image가 아니면 변환
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        # RGB로 변환 (grayscale인 경우 대비)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 성별 정보 (1: 여성 -> 0, 2: 남성 -> 1로 변환하여 이진 분류)
        gender_id = self.data.iloc[idx]['gender']
        gender_label = 0 if gender_id == 1 else 1  # 여성: 0, 남성: 1
        gender_label = torch.tensor(gender_label, dtype=torch.long)
        
        # 이미지 전처리 적용
        if self.transform:
            image = self.transform(image)
        
        return image, gender_label


# 이미지 전처리 정의
# Train용: Data Augmentation 포함
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet은 224x224 입력 사용
    transforms.RandomHorizontalFlip(p=0.5),  # 50% 확률로 좌우 반전
    transforms.RandomRotation(degrees=15),  # ±15도 회전
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 밝기, 대비, 채도 조정
    transforms.ToTensor(),  # Tensor로 변환 [0, 1] 범위
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet 평균/표준편차
                        std=[0.229, 0.224, 0.225])
])

# Validation용: Augmentation 없이 기본 전처리만
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet은 224x224 입력 사용
    transforms.ToTensor(),  # Tensor로 변환 [0, 1] 범위
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet 평균/표준편차
                        std=[0.229, 0.224, 0.225])
])

print("이미지 전처리 설정 완료:")
print("  Train: Resize(224x224) + Augmentation (Flip, Rotation, ColorJitter) + Normalize")
print("  Validation: Resize(224x224) + Normalize")

step2_time = time.time() - step2_start_time
print(f"\n[2단계 완료] 소요 시간: {step2_time:.2f}초")

# ================================================================================
# 3. Train/Validation 데이터 분할
# ================================================================================
print("\n" + "=" * 80)
print("3단계: Train/Validation 데이터 분할")
print("=" * 80)
step3_start_time = time.time()

# 80:20 분할 (성별 비율 유지)
train_df, val_df = train_test_split(
    df_filtered, 
    test_size=0.2, 
    random_state=RANDOM_SEED,
    stratify=df_filtered['gender']  # 성별 비율 유지
)

# 학습에 사용한 원본 데이터셋 인덱스 저장 (평가 시 제외하기 위해)
used_indices = set(train_df['original_index'].tolist() + val_df['original_index'].tolist())
import json
with open('training_indices.json', 'w') as f:
    json.dump(list(used_indices), f)
print(f"\n[데이터 분리 정보] 학습에 사용한 원본 인덱스 {len(used_indices)}개를 'training_indices.json'에 저장했습니다.")
print(f"  이 인덱스들은 evaluate_model.py에서 제외됩니다.")

print(f"\nTrain 데이터: {len(train_df)}개")
print(f"  - 여성: {len(train_df[train_df['gender'] == 1])}명, 남성: {len(train_df[train_df['gender'] == 2])}명")
print(f"Validation 데이터: {len(val_df)}개")
print(f"  - 여성: {len(val_df[val_df['gender'] == 1])}명, 남성: {len(val_df[val_df['gender'] == 2])}명")

# 배치 사이즈 자동 조정 (GPU 메모리 고려)
def adjust_batch_size_for_gpu(data_size, initial_batch_size, force_cpu=False):
    """GPU 메모리를 고려하여 배치 사이즈 조정"""
    if not torch.cuda.is_available() or force_cpu:
        return initial_batch_size
    
    # GPU 메모리 확인
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    free_memory_gb = (torch.cuda.get_device_properties(0).total_memory - 
                     torch.cuda.memory_allocated(0)) / (1024**3)
    
    # 배치당 예상 GPU 메모리 사용량 (224x224 RGB 이미지 기준, 대략적 추정)
    # ResNet18 + 배치: 약 0.5GB per batch (배치 사이즈 32 기준)
    memory_per_batch_gb = 0.5 * (initial_batch_size / 32)
    
    # 안전 여유를 두고 사용 가능한 메모리의 70%만 사용
    safe_gpu_memory = free_memory_gb * 0.7
    
    # GPU 메모리가 부족하면 배치 사이즈 감소
    if memory_per_batch_gb > safe_gpu_memory:
        adjusted_batch = max(1, int(initial_batch_size * (safe_gpu_memory / memory_per_batch_gb)))
        print(f"\n[GPU 메모리 체크]")
        print(f"  GPU 메모리: {gpu_memory_gb:.2f} GB (사용 가능: {free_memory_gb:.2f} GB)")
        print(f"  예상 배치 메모리: {memory_per_batch_gb:.2f} GB")
        print(f"  안전 메모리 한계: {safe_gpu_memory:.2f} GB")
        print(f"  배치 사이즈 조정: {initial_batch_size} → {adjusted_batch}")
        return adjusted_batch
    
    return initial_batch_size

if BATCH_SIZE is None:
    # train 데이터 개수에 따라 자동 조정
    if len(train_df) < 100:
        batch_size = 4
    elif len(train_df) < 500:
        batch_size = 8
    elif len(train_df) < 2000:
        batch_size = 16
    else:
        batch_size = 32
    print(f"배치 사이즈 자동 설정: {batch_size}")
else:
    batch_size = BATCH_SIZE
    print(f"배치 사이즈 고정: {batch_size}")

# GPU 메모리를 고려한 배치 사이즈 조정
batch_size = adjust_batch_size_for_gpu(len(train_df), batch_size, force_cpu=args.cpu)

# Dataset 및 DataLoader 생성 (Train은 Augmentation 적용, Validation은 기본 전처리만)
train_dataset = PeopleGenderDataset(train_df, transform=train_transform)
val_dataset = PeopleGenderDataset(val_df, transform=val_transform)

# DataLoader 설정: GPU 사용 시 pin_memory=True로 설정하여 데이터 전송 최적화
# num_workers는 메모리 사용량을 고려하여 0으로 설정 (필요시 조정 가능)
num_workers = 0  # 멀티프로세싱 시 메모리 사용량 증가하므로 0으로 설정
pin_memory = torch.cuda.is_available() and not args.cpu  # GPU 사용 시에만 pin_memory 활성화

train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin_memory
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=batch_size, 
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory
)

print(f"Train Batch 개수: {len(train_loader)}")
print(f"Validation Batch 개수: {len(val_loader)}")

step3_time = time.time() - step3_start_time
print(f"\n[3단계 완료] 소요 시간: {step3_time:.2f}초")

# ================================================================================
# 4. Transfer Learning 모델 정의 (ResNet18)
# ================================================================================
print("\n" + "=" * 80)
print("4단계: Transfer Learning 모델 정의 (ResNet18)")
print("=" * 80)
step4_start_time = time.time()


def create_resnet_model(num_classes=2, pretrained=True):
    """ResNet18 기반 Transfer Learning 모델 생성"""
    # 사전학습된 ResNet18 로드
    model = models.resnet18(pretrained=pretrained)
    
    # 마지막 fully connected layer를 우리의 분류 작업에 맞게 수정
    # ResNet18의 마지막 FC layer는 1000개 클래스(ImageNet)를 위한 것이므로
    # 2개 클래스(여성/남성)로 변경
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),  # Dropout 추가로 과적합 방지
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    
    return model


# 모델 초기화
# 명령줄 인자에 따라 디바이스 선택
if args.cpu:
    device = torch.device('cpu')
    print("\n[디바이스 설정] CPU로 강제 실행 (--cpu 플래그 사용)")
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print("\n[디바이스 설정] GPU 사용")
    else:
        print("\n[디바이스 설정] CUDA를 사용할 수 없어 CPU로 실행")

model = create_resnet_model(num_classes=2, pretrained=True).to(device)  # 이진 분류: 여성(0) / 남성(1)

if torch.cuda.is_available() and not args.cpu:
    print("사전학습된 ResNet18 모델 로드 완료 (ImageNet 가중치 사용)")
else:
    print("사전학습된 ResNet18 모델 로드 완료 (ImageNet 가중치 사용)")
    if args.cpu:
        print("CPU 모드로 실행 중입니다.")
    else:
        print("주의: CPU로 학습하므로 시간이 오래 걸릴 수 있습니다.")

print(f"모델 초기화 완료")
print(f"사용 디바이스: {device}")
if torch.cuda.is_available() and not args.cpu:
    print(f"GPU 이름: {torch.cuda.get_device_name(0)}")
    print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
elif args.cpu:
    print("CPU 모드로 실행 중입니다.")
else:
    print("경고: CUDA를 사용할 수 없습니다. CPU로 학습합니다.")
print(f"모델 파라미터 개수: {sum(p.numel() for p in model.parameters()):,}")

step4_time = time.time() - step4_start_time
print(f"\n[4단계 완료] 소요 시간: {step4_time:.2f}초")


# ================================================================================
# 5. 손실 함수 및 옵티마이저 설정
# ================================================================================
print("\n" + "=" * 80)
print("5단계: 손실 함수 및 옵티마이저 설정")
print("=" * 80)
step5_start_time = time.time()

# 클래스 가중치 계산 (불균형 데이터 보정)
# 여성(0)과 남성(1)의 비율에 반비례하는 가중치 부여
female_count = len(train_df[train_df['gender'] == 1])  # 여성 개수
male_count = len(train_df[train_df['gender'] == 2])     # 남성 개수
total_count = len(train_df)

# 가중치: 클래스가 적을수록 높은 가중치 (여성에 더 높은 가중치)
class_weights = torch.tensor([
    total_count / (2 * female_count),  # 여성(0) 가중치
    total_count / (2 * male_count)      # 남성(1) 가중치
], dtype=torch.float32).to(device)

print(f"클래스 분포: 여성 {female_count}명, 남성 {male_count}명")
print(f"클래스 가중치: 여성={class_weights[0]:.4f}, 남성={class_weights[1]:.4f}")

# 분류 문제이므로 Cross Entropy Loss 사용 (클래스 가중치 포함)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("손실 함수: Cross Entropy Loss (클래스 가중치 적용)")
print("옵티마이저: Adam (lr=0.001)")

step5_time = time.time() - step5_start_time
print(f"\n[5단계 완료] 소요 시간: {step5_time:.2f}초")

# ================================================================================
# 6. 학습 및 평가 함수 정의
# ================================================================================
print("\n" + "=" * 80)
print("6단계: 학습 및 평가 함수 정의")
print("=" * 80)
step6_start_time = time.time()


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """한 에포크 학습"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # 통계
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_accuracy = 100.0 * correct / total
    
    return epoch_loss, epoch_accuracy


def evaluate(model, dataloader, criterion, device):
    """모델 평가 (클래스별 정확도 포함)"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    # 클래스별 통계 (0: 여성, 1: 남성)
    class_correct = [0, 0]
    class_total = [0, 0]
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 통계
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 클래스별 정확도 계산
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == labels[i]:
                    class_correct[label] += 1
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_accuracy = 100.0 * correct / total
    
    # 클래스별 정확도
    female_acc = 100.0 * class_correct[0] / class_total[0] if class_total[0] > 0 else 0.0
    male_acc = 100.0 * class_correct[1] / class_total[1] if class_total[1] > 0 else 0.0
    
    return epoch_loss, epoch_accuracy, female_acc, male_acc


print("학습 및 평가 함수 정의 완료")

step6_time = time.time() - step6_start_time
print(f"\n[6단계 완료] 소요 시간: {step6_time:.2f}초")

# ================================================================================
# 7. 모델 학습 실행
# ================================================================================
print("\n" + "=" * 80)
print("7단계: 모델 학습 시작")
print("=" * 80)
step7_start_time = time.time()

num_epochs = 30
best_val_accuracy = 0.0
best_val_loss = float('inf')
patience = 5  # Early Stopping: Val Loss가 개선되지 않으면 5 에포크 후 중단
patience_counter = 0

print(f"총 에포크: {num_epochs}")
print(f"배치 크기: {batch_size}")
print(f"Early Stopping Patience: {patience} 에포크")
print("\n학습 시작...\n")

for epoch in range(num_epochs):
    # 학습
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    
    # 검증 (클래스별 정확도 포함)
    val_loss, val_acc, val_female_acc, val_male_acc = evaluate(model, val_loader, criterion, device)
    
    # 최고 모델 저장 (Validation Accuracy 기준)
    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        best_val_loss = val_loss
        patience_counter = 0  # 개선되면 카운터 리셋
        torch.save(model.state_dict(), 'best_gender_model.pth')
    else:
        patience_counter += 1
    
    # GPU 메모리 정리 (메모리 누수 방지)
    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.empty_cache()
    
    # 진행 상황 출력 (5 에포크마다 또는 첫 에포크)
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Accuracy:   {val_acc:.2f}%")
        print(f"    - 여성 정확도: {val_female_acc:.2f}% | 남성 정확도: {val_male_acc:.2f}%")
        print(f"  Best Val Accuracy: {best_val_accuracy:.2f}%")
        if torch.cuda.is_available() and not args.cpu:
            gpu_memory_used = torch.cuda.memory_allocated(0) / (1024**3)
            print(f"  GPU 메모리 사용량: {gpu_memory_used:.2f} GB")
        if patience_counter > 0:
            print(f"  Early Stopping: {patience_counter}/{patience} (개선 없음)")
        print("-" * 60)
    
    # Early Stopping 체크
    if patience_counter >= patience:
        print(f"\nEarly Stopping: {patience} 에포크 동안 개선이 없어 학습을 중단합니다.")
        print(f"최고 성능: Val Accuracy {best_val_accuracy:.2f}% (Epoch {epoch - patience + 1})")
        break

print("\n학습 완료!")
print(f"최고 Validation Accuracy: {best_val_accuracy:.2f}%")
print(f"모델 저장 위치: best_gender_model.pth")

step7_time = time.time() - step7_start_time
print(f"\n[7단계 완료] 소요 시간: {step7_time:.2f}초 ({step7_time/60:.2f}분)")

# ================================================================================
# 8. 테스트 예측 (샘플 확인)
# ================================================================================
print("\n" + "=" * 80)
print("8단계: 샘플 예측 확인")
print("=" * 80)
step8_start_time = time.time()

# 최고 모델 로드
model.load_state_dict(torch.load('best_gender_model.pth', map_location=device))
model.eval()

# Validation 데이터에서 몇 개 샘플 예측
gender_names = {0: "여성", 1: "남성"}

with torch.no_grad():
    for i, (images, labels) in enumerate(val_loader):
        if i >= 1:  # 첫 번째 배치만
            break
        
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
        
        print("\n예측 결과 (첫 번째 배치):")
        print("-" * 60)
        for j in range(min(10, len(labels))):  # 최대 10개만
            actual_gender = gender_names[labels[j].item()]
            predicted_gender = gender_names[predictions[j].item()]
            confidence = torch.softmax(outputs[j], dim=0)[predictions[j]].item() * 100
            is_correct = "✓" if labels[j].item() == predictions[j].item() else "✗"
            print(f"{is_correct} 실제: {actual_gender:3s} | 예측: {predicted_gender:3s} | "
                  f"신뢰도: {confidence:.1f}%")

step8_time = time.time() - step8_start_time
print(f"\n[8단계 완료] 소요 시간: {step8_time:.2f}초")

# 전체 실행 시간 계산
total_time = time.time() - total_start_time

print("\n" + "=" * 80)
print("모든 작업 완료!")
print("=" * 80)
print(f"\n[전체 실행 시간 요약]")
print(f"  디바이스: {device}")
print(f"  1단계 (데이터 로드 및 전처리): {step1_time:.2f}초 ({step1_time/60:.2f}분)")
print(f"  2단계 (Dataset 클래스 정의): {step2_time:.2f}초")
print(f"  3단계 (Train/Validation 분할): {step3_time:.2f}초")
print(f"  4단계 (모델 정의): {step4_time:.2f}초")
print(f"  5단계 (손실 함수 및 옵티마이저): {step5_time:.2f}초")
print(f"  6단계 (학습/평가 함수 정의): {step6_time:.2f}초")
print(f"  7단계 (모델 학습): {step7_time:.2f}초 ({step7_time/60:.2f}분)")
print(f"  8단계 (샘플 예측): {step8_time:.2f}초")
print(f"  ─────────────────────────────────────────────")
print(f"  총 실행 시간: {total_time:.2f}초 ({total_time/60:.2f}분)")
print("=" * 80)


