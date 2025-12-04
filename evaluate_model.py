"""
학습된 모델을 사용하여 랜덤 샘플 평가 스크립트
"""
from datasets import load_dataset
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random
import json
import os

# ================================================================================
# 설정 변수
# ================================================================================
MODEL_PATH = 'best_gender_model.pth'  # 평가할 모델 파일 경로
NUM_TEST_SAMPLES = 50  # 평가할 랜덤 샘플 개수 (원하는 숫자로 변경 가능: 100, 500, 1000 등)
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

# 학습에 사용한 인덱스 로드 (있는 경우)
training_indices = set()
training_indices_file = 'training_indices.json'
if os.path.exists(training_indices_file):
    with open(training_indices_file, 'r') as f:
        training_indices = set(json.load(f))
    print(f"[데이터 분리] 학습에 사용한 인덱스 {len(training_indices)}개를 제외합니다.")
else:
    print(f"[경고] '{training_indices_file}' 파일을 찾을 수 없습니다.")
    print(f"  학습 데이터와 평가 데이터가 겹칠 수 있습니다.")
    print(f"  people_gender.py를 먼저 실행하여 학습을 완료하세요.")

# 학습에 사용하지 않은 인덱스만 사용
available_indices = [i for i in range(total_size) if i not in training_indices]
print(f"평가에 사용 가능한 인덱스: {len(available_indices):,}개")

if len(available_indices) == 0:
    raise ValueError("평가에 사용할 수 있는 데이터가 없습니다! (모든 데이터가 학습에 사용됨)")

# 메모리 효율성을 위해 충분한 샘플을 먼저 선택
# 필터링 후에도 NUM_TEST_SAMPLES만큼 남도록 여유있게 샘플링
# (사망한 사람, 성별 미지정 등을 고려하여 3-4배 정도 샘플링)
SAMPLE_MULTIPLIER = 4  # 필터링 후 손실을 고려한 배수
sample_size = max(NUM_TEST_SAMPLES * SAMPLE_MULTIPLIER, 1000)  # 최소 1000개

if sample_size < len(available_indices):
    print(f"메모리 효율성을 위해 먼저 {sample_size}개 샘플링 중...")
    # 사용 가능한 인덱스에서 랜덤 샘플링
    random.shuffle(available_indices)
    selected_indices = available_indices[:sample_size]
    selected_indices.sort()  # 정렬하여 순차 접근으로 성능 향상
    # 선택된 인덱스만 로드하여 DataFrame 생성
    df = pd.DataFrame(ds['train'].select(selected_indices))
    # 원본 인덱스 추가
    df['original_index'] = selected_indices
    print(f"샘플링된 데이터 개수: {len(df)}개")
else:
    print("사용 가능한 전체 데이터셋을 DataFrame으로 변환 중... (시간이 걸릴 수 있습니다)")
    df = pd.DataFrame(ds['train'].select(available_indices))
    df['original_index'] = available_indices
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

gender_names = {0: "여성", 1: "남성"}  # 콘솔 출력용 (한글)
gender_names_en = {0: "Female", 1: "Male"}  # 이미지 표시용 (영어)
correct = 0
total = 0
class_correct = [0, 0]  # [여성, 남성]
class_total = [0, 0]
all_predictions = []
all_labels = []
all_images = []  # 원본 이미지 저장
all_confidences = []  # 신뢰도 저장

print("\n예측 결과:")
print("-" * 80)

# 원본 이미지를 DataFrame에서 가져오는 함수
def get_original_image(idx):
    """DataFrame에서 원본 이미지 가져오기"""
    image = test_df.iloc[idx]['image']
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image

with torch.no_grad():
    sample_idx = 0  # 전체 샘플 인덱스 추적
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
            # DataFrame에서 원본 이미지 가져오기
            all_images.append(get_original_image(sample_idx))
            all_confidences.append(confidence)
            
            # 결과 출력 (모든 샘플)
            print(f"{symbol} 실제: {actual_gender:3s} | 예측: {predicted_gender:3s} | "
                  f"신뢰도: {confidence:.1f}%")
            
            sample_idx += 1

def show_predictions(images, labels, predictions, confidences, num_cols=5, img_size=224):
    """예측 결과와 함께 이미지를 그리드 형태로 표시 (PIL 사용)"""
    num_images = len(images)
    num_rows = (num_images + num_cols - 1) // num_cols
    
    # 각 셀 크기 계산 (이미지 + 텍스트 영역)
    text_height = 60  # 텍스트 영역 높이
    cell_width = img_size
    cell_height = img_size + text_height
    
    # 전체 그리드 이미지 생성
    grid_width = num_cols * cell_width
    grid_height = num_rows * cell_height
    grid_image = Image.new('RGB', (grid_width, grid_height), color='white')
    
    # 폰트 로드 시도 (시스템 폰트 사용)
    try:
        # Linux에서 일반적인 폰트 경로들 시도
        font_paths = [
            '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
            '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
            '/System/Library/Fonts/Helvetica.ttc',  # macOS
        ]
        font = None
        for path in font_paths:
            try:
                font = ImageFont.truetype(path, 16)
                break
            except:
                continue
        if font is None:
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    
    # 각 이미지를 그리드에 배치
    for idx in range(num_images):
        row = idx // num_cols
        col = idx % num_cols
        
        # 이미지 위치 계산
        x = col * cell_width
        y = row * cell_height
        
        # 이미지 리사이즈 및 붙여넣기
        img = images[idx].copy()
        img = img.resize((img_size, img_size), Image.Resampling.LANCZOS)
        grid_image.paste(img, (x, y))
        
        # 텍스트 영역에 정보 표시
        draw = ImageDraw.Draw(grid_image)
        text_y = y + img_size + 5
        
        # 실제/예측/신뢰도 정보 (영어로 표시)
        actual = gender_names_en[labels[idx]]
        predicted = gender_names_en[predictions[idx]]
        conf = confidences[idx]
        is_correct = labels[idx] == predictions[idx]
        
        symbol = "✓" if is_correct else "✗"
        text_color = (0, 200, 0) if is_correct else (200, 0, 0)  # 초록색 또는 빨간색
        
        # 텍스트 그리기 (영어)
        text_line1 = f"{symbol} Actual: {actual} | Predicted: {predicted}"
        text_line2 = f"Confidence: {conf:.1f}%"
        
        # 텍스트 배경 (가독성 향상)
        text_bbox1 = draw.textbbox((x + 5, text_y), text_line1, font=font)
        text_bbox2 = draw.textbbox((x + 5, text_y + 20), text_line2, font=font)
        
        # 반투명 배경
        overlay = Image.new('RGBA', (cell_width, text_height), (255, 255, 255, 200))
        grid_image.paste(overlay, (x, y + img_size), overlay)
        
        # 텍스트 그리기
        draw.text((x + 5, text_y), text_line1, fill=text_color, font=font)
        draw.text((x + 5, text_y + 20), text_line2, fill=(0, 0, 0), font=font)
    
    # 결과 이미지 저장
    grid_image.save('evaluation_results.png', 'PNG', quality=95)
    print("\n이미지 결과가 'evaluation_results.png' 파일로 저장되었습니다.")
    
    return grid_image

# 이미지 표시
print("\n이미지 결과 표시 중...")
show_predictions(all_images, all_labels, all_predictions, all_confidences)

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


