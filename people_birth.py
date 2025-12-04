from datasets import load_dataset
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import random

# ================================================================================
# 설정 변수
# ================================================================================
NUM_SAMPLES = 1000  # None: 전체 데이터 사용, 숫자: 해당 개수만큼만 사용 (예: 100, 500, 1000)
BATCH_SIZE = None   # None: 자동 조정, 숫자: 고정 배치 사이즈 (예: 8, 16, 32, 64)
RANDOM_SEED = 42    # 재현성을 위한 랜덤 시드

# ================================================================================
# 1. 데이터 로드 및 전처리
# ================================================================================
print("=" * 80)
print("1단계: 데이터셋 로드 및 전처리")
print("=" * 80)

# 랜덤 시드 설정 (재현성)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# 데이터셋 로드
print("데이터셋 로드 중...")
ds = load_dataset("ashraq/tmdb-people-image")
total_size = len(ds['train'])
print(f"전체 데이터셋 크기: {total_size:,}개")

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
    print(f"샘플링된 데이터 개수: {len(df)}개")
else:
    # 전체 데이터 사용 시에도 배치로 처리하여 메모리 효율성 향상
    print("전체 데이터셋을 DataFrame으로 변환 중... (시간이 걸릴 수 있습니다)")
    df = pd.DataFrame(ds['train'])
    print(f"전체 데이터 개수: {len(df)}개")

# 데이터 셔플 (분할 전에 셔플 수행)
if NUM_SAMPLES is None or NUM_SAMPLES >= total_size:
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


# ================================================================================
# 2. PyTorch Dataset 클래스 정의
# ================================================================================
print("\n" + "=" * 80)
print("2단계: PyTorch Dataset 클래스 정의")
print("=" * 80)


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


# 이미지 전처리 정의 (간단한 CNN용)
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # 128x128로 리사이즈
    transforms.ToTensor(),  # Tensor로 변환 [0, 1] 범위
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet 평균/표준편차
                        std=[0.229, 0.224, 0.225])
])

print("이미지 전처리 설정 완료: Resize(128x128) -> ToTensor -> Normalize")


# ================================================================================
# 3. Train/Validation 데이터 분할
# ================================================================================
print("\n" + "=" * 80)
print("3단계: Train/Validation 데이터 분할")
print("=" * 80)

# 80:20 분할 (성별 비율 유지)
train_df, val_df = train_test_split(
    df_filtered, 
    test_size=0.2, 
    random_state=RANDOM_SEED,
    stratify=df_filtered['gender']  # 성별 비율 유지
)

print(f"Train 데이터: {len(train_df)}개")
print(f"  - 여성: {len(train_df[train_df['gender'] == 1])}명, 남성: {len(train_df[train_df['gender'] == 2])}명")
print(f"Validation 데이터: {len(val_df)}개")
print(f"  - 여성: {len(val_df[val_df['gender'] == 1])}명, 남성: {len(val_df[val_df['gender'] == 2])}명")

# 배치 사이즈 자동 조정
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

# Dataset 및 DataLoader 생성
train_dataset = PeopleGenderDataset(train_df, transform=transform)
val_dataset = PeopleGenderDataset(val_df, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f"Train Batch 개수: {len(train_loader)}")
print(f"Validation Batch 개수: {len(val_loader)}")


# ================================================================================
# 4. 간단한 CNN 모델 정의
# ================================================================================
print("\n" + "=" * 80)
print("4단계: CNN 모델 정의")
print("=" * 80)


class SimpleCNN(nn.Module):
    """성별 분류를 위한 간단한 CNN 모델"""
    
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 128x128 -> 128x128
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 64x64 -> 64x64
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 32x32 -> 32x32
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)  # 크기를 절반으로 줄임
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 16 * 16, 256)  # 16x16은 3번의 pooling 결과
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, num_classes)  # 성별 분류 (이진 분류: 여성/남성)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
        # Activation
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Conv1 + Pool: 128x128 -> 64x64
        x = self.pool(self.relu(self.conv1(x)))
        
        # Conv2 + Pool: 64x64 -> 32x32
        x = self.pool(self.relu(self.conv2(x)))
        
        # Conv3 + Pool: 32x32 -> 16x16
        x = self.pool(self.relu(self.conv3(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x  # (batch_size, num_classes) 형태


# 모델 초기화
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN(num_classes=2).to(device)  # 이진 분류: 여성(0) / 남성(1)

print(f"모델 초기화 완료")
print(f"사용 디바이스: {device}")
if torch.cuda.is_available():
    print(f"GPU 이름: {torch.cuda.get_device_name(0)}")
    print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("경고: CUDA를 사용할 수 없습니다. CPU로 학습합니다.")
print(f"모델 파라미터 개수: {sum(p.numel() for p in model.parameters()):,}")


# ================================================================================
# 5. 손실 함수 및 옵티마이저 설정
# ================================================================================
print("\n" + "=" * 80)
print("5단계: 손실 함수 및 옵티마이저 설정")
print("=" * 80)

# 분류 문제이므로 Cross Entropy Loss 사용
criterion = nn.CrossEntropyLoss()

# Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("손실 함수: Cross Entropy Loss (분류)")
print("옵티마이저: Adam (lr=0.001)")


# ================================================================================
# 6. 학습 및 평가 함수 정의
# ================================================================================
print("\n" + "=" * 80)
print("6단계: 학습 및 평가 함수 정의")
print("=" * 80)


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
    """모델 평가"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
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
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_accuracy = 100.0 * correct / total
    
    return epoch_loss, epoch_accuracy


print("학습 및 평가 함수 정의 완료")


# ================================================================================
# 7. 모델 학습 실행
# ================================================================================
print("\n" + "=" * 80)
print("7단계: 모델 학습 시작")
print("=" * 80)

num_epochs = 30
best_val_accuracy = 0.0

print(f"총 에포크: {num_epochs}")
print(f"배치 크기: {batch_size}")
print("\n학습 시작...\n")

for epoch in range(num_epochs):
    # 학습
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    
    # 검증
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    
    # 최고 모델 저장
    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        torch.save(model.state_dict(), 'best_gender_model.pth')
    
    # 진행 상황 출력 (5 에포크마다)
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Accuracy:   {val_acc:.2f}%")
        print(f"  Best Val Accuracy: {best_val_accuracy:.2f}%")
        print("-" * 60)

print("\n학습 완료!")
print(f"최고 Validation Accuracy: {best_val_accuracy:.2f}%")
print(f"모델 저장 위치: best_gender_model.pth")


# ================================================================================
# 8. 테스트 예측 (샘플 확인)
# ================================================================================
print("\n" + "=" * 80)
print("8단계: 샘플 예측 확인")
print("=" * 80)

# 최고 모델 로드
model.load_state_dict(torch.load('best_gender_model.pth'))
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

print("\n" + "=" * 80)
print("모든 작업 완료!")
print("=" * 80)


