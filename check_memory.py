"""
메모리 사용량 확인 스크립트
"""
import psutil
import os
import sys
from datasets import load_dataset
import pandas as pd
import torch

def get_memory_usage():
    """현재 메모리 사용량 반환 (GB)"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024**3)  # GB 단위

def get_system_memory():
    """시스템 전체 메모리 정보 반환"""
    mem = psutil.virtual_memory()
    return {
        'total': mem.total / (1024**3),  # GB
        'available': mem.available / (1024**3),  # GB
        'used': mem.used / (1024**3),  # GB
        'percent': mem.percent
    }

print("=" * 80)
print("메모리 사용량 확인")
print("=" * 80)

# 시스템 메모리 정보
sys_mem = get_system_memory()
print(f"\n[시스템 메모리 정보]")
print(f"전체 메모리: {sys_mem['total']:.2f} GB")
print(f"사용 가능: {sys_mem['available']:.2f} GB")
print(f"사용 중: {sys_mem['used']:.2f} GB")
print(f"사용률: {sys_mem['percent']:.1f}%")

# 현재 프로세스 메모리
initial_mem = get_memory_usage()
print(f"\n[현재 프로세스 메모리]")
print(f"초기 메모리 사용량: {initial_mem:.2f} GB")

# 데이터셋 로드 전후 메모리 비교
print(f"\n[데이터셋 로드 테스트]")
print("-" * 80)

# 1. 데이터셋 로드 (메타데이터만)
print("\n1. 데이터셋 메타데이터 로드 중...")
before_load = get_memory_usage()
ds = load_dataset("ashraq/tmdb-people-image")
after_load = get_memory_usage()
print(f"   메모리 사용량: {before_load:.2f} GB → {after_load:.2f} GB")
print(f"   증가량: {after_load - before_load:.4f} GB")

# 2. 100개 샘플 로드
print("\n2. 100개 샘플 로드 중...")
before_100 = get_memory_usage()
indices_100 = list(range(100))
df_100 = pd.DataFrame(ds['train'].select(indices_100))
after_100 = get_memory_usage()
mem_100 = after_100 - before_100
print(f"   메모리 사용량: {before_100:.2f} GB → {after_100:.2f} GB")
print(f"   증가량: {mem_100:.4f} GB")
print(f"   샘플당 평균: {mem_100 / 100 * 1024:.2f} MB")

# 3. 1000개 샘플 로드
print("\n3. 1000개 샘플 로드 중...")
before_1000 = get_memory_usage()
indices_1000 = list(range(1000))
df_1000 = pd.DataFrame(ds['train'].select(indices_1000))
after_1000 = get_memory_usage()
mem_1000 = after_1000 - before_1000
print(f"   메모리 사용량: {before_1000:.2f} GB → {after_1000:.2f} GB")
print(f"   증가량: {mem_1000:.4f} GB")
print(f"   샘플당 평균: {mem_1000 / 1000 * 1024:.2f} MB")

# 4. DataFrame 크기 정보
print("\n[DataFrame 크기 정보]")
print(f"100개 샘플 DataFrame 크기: {df_100.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
print(f"1000개 샘플 DataFrame 크기: {df_1000.memory_usage(deep=True).sum() / (1024**2):.2f} MB")

# 5. 이미지 데이터 크기 추정
print("\n[이미지 데이터 크기 추정]")
if 'image' in df_100.columns:
    # 첫 번째 이미지 크기 확인
    first_image = df_100.iloc[0]['image']
    if hasattr(first_image, 'size'):
        img_size = first_image.size
        # RGB 이미지라고 가정 (3 채널)
        estimated_size_per_image = img_size[0] * img_size[1] * 3 / (1024**2)  # MB
        print(f"   이미지 크기 (예상): {img_size[0]}x{img_size[1]}")
        print(f"   이미지당 메모리 (예상): {estimated_size_per_image:.2f} MB")
        print(f"   100개 이미지 예상 메모리: {estimated_size_per_image * 100:.2f} MB")
        print(f"   1000개 이미지 예상 메모리: {estimated_size_per_image * 1000 / 1024:.2f} GB")

# 6. 예상 메모리 사용량
print("\n[예상 메모리 사용량]")
sample_per_gb = 1000 / mem_1000 if mem_1000 > 0 else 0
print(f"   GB당 샘플 수: {sample_per_gb:.0f}개")
print(f"   1000개 샘플 예상 메모리: {mem_1000:.4f} GB ({mem_1000 * 1024:.2f} MB)")
print(f"   10000개 샘플 예상 메모리: {mem_1000 * 10:.4f} GB ({mem_1000 * 10 * 1024:.2f} MB)")

# 7. 권장 사항
print("\n[권장 사항]")
available_gb = sys_mem['available']
safe_limit = available_gb * 0.5  # 사용 가능 메모리의 50%까지 사용
recommended_samples = int(safe_limit * sample_per_gb) if sample_per_gb > 0 else 1000
print(f"   사용 가능 메모리: {available_gb:.2f} GB")
print(f"   안전한 사용 한계 (50%): {safe_limit:.2f} GB")
print(f"   권장 샘플 수: {recommended_samples:,}개 이하")

print("\n" + "=" * 80)
print("메모리 확인 완료!")
print("=" * 80)

