from datasets import load_dataset
import pandas as pd
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

# ================================================================================
# 1. 데이터 로드 및 전처리
# ================================================================================
print("=" * 80)
print("1단계: 데이터셋 로드 및 나이 계산")
print("=" * 80)

# 데이터셋 로드
ds = load_dataset("ashraq/tmdb-people-image")

# 100개 샘플을 DataFrame으로 변환
df = pd.DataFrame(ds['train'][:100])

print(f"전체 데이터 개수: {len(df)}")
print(f"컬럼 목록: {df.columns.tolist()}")


# 나이 계산 함수
def calculate_age(birthday_str, reference_year=2025):
    """생년월일로부터 나이를 계산"""
    if pd.isna(birthday_str) or birthday_str == '' or birthday_str is None:
        return None
    
    try:
        birth_date = pd.to_datetime(birthday_str)
        age = reference_year - birth_date.year
        return age
    except:
        return None


# 생년월일 컬럼 찾기
birthday_columns = [col for col in df.columns if 'birth' in col.lower() or 'dob' in col.lower()]

if birthday_columns:
    birthday_col = birthday_columns[0]
    df['age_2025'] = df[birthday_col].apply(calculate_age)
    
    # 나이 정보가 있는 데이터만 필터링
    df_filtered = df[df['age_2025'].notna()].copy()
    
    print(f"\n생년월일 컬럼: {birthday_col}")
    print(f"나이 정보가 있는 데이터: {len(df_filtered)}개")
    print(f"나이 범위: {df_filtered['age_2025'].min():.0f} ~ {df_filtered['age_2025'].max():.0f}세")
    print(f"평균 나이: {df_filtered['age_2025'].mean():.1f}세")
else:
    raise ValueError("생년월일 컬럼을 찾을 수 없습니다!")


# ================================================================================
# 2. PyTorch Dataset 클래스 정의
# ================================================================================
print("\n" + "=" * 80)
print("2단계: PyTorch Dataset 클래스 정의")
print("=" * 80)


class PeopleAgeDataset(Dataset):
    """TMDB 인물 이미지와 나이를 위한 Dataset 클래스"""
    
    def __init__(self, data, transform=None):
        """
        Args:
            data: DataFrame (image, age_2025 컬럼 포함)
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
        
        # 나이 정보
        age = float(self.data.iloc[idx]['age_2025'])
        
        # 이미지 전처리 적용
        if self.transform:
            image = self.transform(image)
        
        return image, age


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

# 80:20 분할
train_df, val_df = train_test_split(df_filtered, test_size=0.2, random_state=42)

print(f"Train 데이터: {len(train_df)}개")
print(f"Validation 데이터: {len(val_df)}개")

# Dataset 및 DataLoader 생성
train_dataset = PeopleAgeDataset(train_df, transform=transform)
val_dataset = PeopleAgeDataset(val_df, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

print(f"Train Batch 개수: {len(train_loader)}")
print(f"Validation Batch 개수: {len(val_loader)}")


# ================================================================================
# 4. 간단한 CNN 모델 정의
# ================================================================================
print("\n" + "=" * 80)
print("4단계: CNN 모델 정의")
print("=" * 80)


class SimpleCNN(nn.Module):
    """나이 예측을 위한 간단한 CNN 모델"""
    
    def __init__(self):
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
        self.fc3 = nn.Linear(64, 1)  # 나이 예측 (회귀)
        
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
        
        return x.squeeze()  # (batch_size,) 형태로 변환


# 모델 초기화
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN().to(device)

print(f"모델 초기화 완료")
print(f"사용 디바이스: {device}")
print(f"모델 파라미터 개수: {sum(p.numel() for p in model.parameters()):,}")


# ================================================================================
# 5. 손실 함수 및 옵티마이저 설정
# ================================================================================
print("\n" + "=" * 80)
print("5단계: 손실 함수 및 옵티마이저 설정")
print("=" * 80)

# 회귀 문제이므로 MSE Loss 사용
criterion = nn.MSELoss()

# Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("손실 함수: MSE (Mean Squared Error)")
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
    running_mae = 0.0
    
    for images, ages in dataloader:
        images = images.to(device)
        ages = ages.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, ages)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # 통계
        running_loss += loss.item() * images.size(0)
        running_mae += torch.abs(outputs - ages).sum().item()
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_mae = running_mae / len(dataloader.dataset)
    
    return epoch_loss, epoch_mae


def evaluate(model, dataloader, criterion, device):
    """모델 평가"""
    model.eval()
    running_loss = 0.0
    running_mae = 0.0
    
    with torch.no_grad():
        for images, ages in dataloader:
            images = images.to(device)
            ages = ages.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, ages)
            
            # 통계
            running_loss += loss.item() * images.size(0)
            running_mae += torch.abs(outputs - ages).sum().item()
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_mae = running_mae / len(dataloader.dataset)
    
    return epoch_loss, epoch_mae


print("학습 및 평가 함수 정의 완료")


# ================================================================================
# 7. 모델 학습 실행
# ================================================================================
print("\n" + "=" * 80)
print("7단계: 모델 학습 시작")
print("=" * 80)

num_epochs = 30
best_val_mae = float('inf')

print(f"총 에포크: {num_epochs}")
print(f"배치 크기: 8")
print("\n학습 시작...\n")

for epoch in range(num_epochs):
    # 학습
    train_loss, train_mae = train_one_epoch(model, train_loader, criterion, optimizer, device)
    
    # 검증
    val_loss, val_mae = evaluate(model, val_loader, criterion, device)
    
    # 최고 모델 저장
    if val_mae < best_val_mae:
        best_val_mae = val_mae
        torch.save(model.state_dict(), 'best_age_model.pth')
    
    # 진행 상황 출력 (5 에포크마다)
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {train_loss:.4f} | Train MAE: {train_mae:.2f}세")
        print(f"  Val Loss:   {val_loss:.4f} | Val MAE:   {val_mae:.2f}세")
        print(f"  Best Val MAE: {best_val_mae:.2f}세")
        print("-" * 60)

print("\n학습 완료!")
print(f"최고 Validation MAE: {best_val_mae:.2f}세")
print(f"모델 저장 위치: best_age_model.pth")


# ================================================================================
# 8. 테스트 예측 (샘플 확인)
# ================================================================================
print("\n" + "=" * 80)
print("8단계: 샘플 예측 확인")
print("=" * 80)

# 최고 모델 로드
model.load_state_dict(torch.load('best_age_model.pth'))
model.eval()

# Validation 데이터에서 몇 개 샘플 예측
with torch.no_grad():
    for i, (images, ages) in enumerate(val_loader):
        if i >= 1:  # 첫 번째 배치만
            break
        
        images = images.to(device)
        ages = ages.to(device)
        
        predictions = model(images)
        
        print("\n예측 결과 (첫 번째 배치):")
        print("-" * 60)
        for j in range(min(5, len(ages))):  # 최대 5개만
            print(f"실제 나이: {ages[j].item():.1f}세 | 예측 나이: {predictions[j].item():.1f}세 | "
                  f"오차: {abs(ages[j].item() - predictions[j].item()):.1f}세")

print("\n" + "=" * 80)
print("모든 작업 완료!")
print("=" * 80)


