"""
로컬 이미지 파일에 대한 성별 평가 스크립트
real_data 폴더의 실제 사진들을 평가합니다.
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob

# ================================================================================
# 설정 변수
# ================================================================================
MODEL_PATH = 'best_gender_model.pth'  # 학습된 모델 파일 경로
IMAGE_DIR = 'real_data'  # 평가할 이미지가 있는 폴더 경로
BATCH_SIZE = 8  # 배치 크기
IMAGE_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']  # 지원 이미지 확장자

# ================================================================================
# 1. Dataset 클래스 정의 (로컬 이미지 파일용)
# ================================================================================
class LocalImageDataset(Dataset):
    """로컬 이미지 파일을 위한 Dataset 클래스"""
    
    def __init__(self, image_paths, transform=None):
        """
        Args:
            image_paths: 이미지 파일 경로 리스트
            transform: 이미지 변환 (torchvision.transforms)
        """
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 이미지 파일 로드
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path)
        except Exception as e:
            print(f"경고: 이미지 로드 실패 ({image_path}): {e}")
            # 실패 시 빈 이미지 생성
            image = Image.new('RGB', (128, 128), color='black')
        
        # RGB로 변환 (grayscale, RGBA 등 처리)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 이미지 전처리 적용
        if self.transform:
            image = self.transform(image)
        
        # 파일명도 반환 (결과 표시용)
        filename = os.path.basename(image_path)
        
        return image, filename


# ================================================================================
# 2. 모델 정의 (evaluate_model.py와 동일한 ResNet 구조)
# ================================================================================
from torchvision import models

def create_resnet_model(num_classes=2, pretrained=False):
    """ResNet18 기반 모델 생성 (evaluate_model.py와 동일)"""
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


# ================================================================================
# 3. 메인 실행 코드
# ================================================================================
print("=" * 80)
print("로컬 이미지 성별 평가")
print("=" * 80)

# 이미지 파일 경로 수집
print(f"\n이미지 폴더: {IMAGE_DIR}")
image_paths = []
for ext in IMAGE_EXTENSIONS:
    image_paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))
    image_paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext.upper())))

# 중복 제거 (Windows에서 대소문자 구분 없이 중복 인식되는 문제 해결)
image_paths = list(set([os.path.normpath(path) for path in image_paths]))
image_paths.sort()  # 정렬

if len(image_paths) == 0:
    raise ValueError(f"'{IMAGE_DIR}' 폴더에서 이미지 파일을 찾을 수 없습니다!")

print(f"발견된 이미지 파일: {len(image_paths)}개")
for i, path in enumerate(image_paths, 1):
    print(f"  {i}. {os.path.basename(path)}")

# 이미지 전처리 (ResNet은 224x224 입력 사용)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet은 224x224 입력 사용
    transforms.ToTensor(),  # Tensor로 변환 [0, 1] 범위
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet 평균/표준편차
                        std=[0.229, 0.224, 0.225])
])

# Dataset 및 DataLoader 생성
dataset = LocalImageDataset(image_paths, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"\n배치 개수: {len(dataloader)}")

# 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용 디바이스: {device}")

# 모델 로드
print(f"\n모델 파일 로드 중: {MODEL_PATH}")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}")

model = create_resnet_model(num_classes=2, pretrained=False).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print("모델 로드 완료!")

# 평가 실행
print("\n" + "=" * 80)
print("예측 결과")
print("=" * 80)

gender_names = {0: "여성", 1: "남성"}  # 콘솔 출력용 (한글)
gender_names_en = {0: "Female", 1: "Male"}  # 이미지 표시용 (영어)
all_predictions = []
all_filenames = []
all_confidences = []
all_images = []  # 원본 이미지 저장용

with torch.no_grad():
    for images, filenames in dataloader:
        images = images.to(device)
        
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
        
        # 각 샘플에 대해 결과 출력
        for i in range(len(filenames)):
            pred = predictions[i].item()
            confidence = torch.softmax(outputs[i], dim=0)[pred].item() * 100
            filename = filenames[i]
            
            predicted_gender = gender_names[pred]  # 콘솔 출력용
            
            all_predictions.append(pred)
            all_filenames.append(filename)
            all_confidences.append(confidence)
            
            # 원본 이미지 저장 (표시용)
            try:
                img = Image.open(os.path.join(IMAGE_DIR, filename))
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                all_images.append(img)
            except Exception as e:
                print(f"경고: 원본 이미지 로드 실패 ({filename}): {e}")
                all_images.append(None)
            
            print(f"파일: {filename:30s} | 예측: {predicted_gender:3s} | 신뢰도: {confidence:.1f}%")

# 결과 요약
print("\n" + "=" * 80)
print("결과 요약")
print("=" * 80)

female_count = sum(1 for p in all_predictions if p == 0)
male_count = sum(1 for p in all_predictions if p == 1)

print(f"\n총 평가 이미지: {len(all_predictions)}개")
print(f"  - 여성으로 예측: {female_count}개")
print(f"  - 남성으로 예측: {male_count}개")

# 평균 신뢰도 계산
avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
print(f"\n평균 신뢰도: {avg_confidence:.1f}%")

# 이미지 결과 표시 (선택사항)
print("\n" + "=" * 80)
print("이미지 결과 저장")
print("=" * 80)

try:
    from PIL import ImageDraw, ImageFont
    
    def create_result_image(images, filenames, predictions, confidences, num_cols=3, img_size=256):
        """예측 결과와 함께 이미지를 그리드 형태로 표시"""
        num_images = len(images)
        if num_images == 0:
            return None
            
        num_rows = (num_images + num_cols - 1) // num_cols
        
        # 각 셀 크기 계산 (이미지 + 텍스트 영역)
        text_height = 80  # 텍스트 영역 높이
        cell_width = img_size
        cell_height = img_size + text_height
        
        # 전체 그리드 이미지 생성
        grid_width = num_cols * cell_width
        grid_height = num_rows * cell_height
        grid_image = Image.new('RGB', (grid_width, grid_height), color='white')
        
        # 폰트 로드 시도
        try:
            font_paths = [
                '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
                '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
            ]
            font = None
            for path in font_paths:
                try:
                    font = ImageFont.truetype(path, 14)
                    break
                except:
                    continue
            if font is None:
                font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()
        
        # 각 이미지를 그리드에 배치
        for idx in range(num_images):
            if images[idx] is None:
                continue
                
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
            
            # 예측/신뢰도 정보 (영어로 표시)
            predicted = gender_names_en[predictions[idx]]  # 영어 사용
            conf = confidences[idx]
            
            # 텍스트 그리기 (영어)
            text_line1 = f"File: {filenames[idx][:20]}"
            text_line2 = f"Predicted: {predicted} ({conf:.1f}%)"
            
            # 반투명 배경
            overlay = Image.new('RGBA', (cell_width, text_height), (255, 255, 255, 220))
            grid_image.paste(overlay, (x, y + img_size), overlay)
            
            # 텍스트 그리기
            draw.text((x + 5, text_y), text_line1, fill=(0, 0, 0), font=font)
            draw.text((x + 5, text_y + 20), text_line2, fill=(0, 100, 200), font=font)
        
        return grid_image
    
    # 결과 이미지 생성 및 저장
    result_image = create_result_image(all_images, all_filenames, all_predictions, all_confidences)
    if result_image:
        output_path = 'real_data_evaluation_results.png'
        result_image.save(output_path, 'PNG', quality=95)
        print(f"이미지 결과가 '{output_path}' 파일로 저장되었습니다.")
    else:
        print("이미지 결과를 생성할 수 없습니다.")
        
except Exception as e:
    print(f"이미지 결과 저장 중 오류 발생: {e}")
    print("텍스트 결과만 출력되었습니다.")

print("\n" + "=" * 80)
print("평가 완료!")
print("=" * 80)

