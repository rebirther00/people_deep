"""
구글 이미지에서 사람 얼굴 사진을 랜덤으로 다운로드하는 스크립트
real_data 폴더에 100장의 이미지를 다운로드합니다.
"""
import os
import time
import random
import hashlib
import requests
from urllib.parse import urlparse, parse_qs
from PIL import Image
from io import BytesIO
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

# ================================================================================
# 설정 변수
# ================================================================================
DOWNLOAD_COUNT = 20  # 다운로드할 이미지 개수
OUTPUT_DIR = 'real_data'  # 이미지 저장 폴더
DELAY_BETWEEN_REQUESTS = 1.5  # 요청 간 딜레이 (초)
SCROLL_PAUSE_TIME = 2  # 스크롤 후 대기 시간 (초)
MAX_SCROLLS = 5  # 최대 스크롤 횟수
HEADLESS_MODE = True  # 브라우저 숨김 모드

# 다양한 검색어 리스트 (다양성 확보)
SEARCH_QUERIES = [
    'portrait', 'face', 'headshot', 'person', 'people',
    'young person', 'adult person', 'elderly person',
    'smiling person', 'professional portrait', 'casual portrait',
    'man portrait', 'woman portrait', 'person face',
    'human face', 'portrait photography', 'headshot photography',
    'person smiling', 'person looking at camera', 'close up face',
    'portrait photo', 'face close up', 'person portrait',
    'professional headshot', 'casual headshot', 'portrait image',
    'face photo', 'person photo', 'human portrait',
    'portrait picture', 'face picture', 'person picture'
]

# ================================================================================
# 1. 폴더 생성 및 초기화
# ================================================================================
def setup_output_directory():
    """출력 폴더 생성"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"폴더 생성: {OUTPUT_DIR}")
    else:
        print(f"기존 폴더 사용: {OUTPUT_DIR}")


# ================================================================================
# 2. Selenium WebDriver 설정
# ================================================================================
def setup_selenium_driver():
    """Chrome WebDriver 설정 및 반환"""
    chrome_options = Options()
    
    if HEADLESS_MODE:
        chrome_options.add_argument('--headless')
    
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
    
    # ChromeDriver 자동 설치 및 설정
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    return driver


# ================================================================================
# 3. 구글 이미지 검색 및 URL 수집
# ================================================================================
def search_google_images(driver, query, max_images=20):
    """
    구글 이미지 검색에서 이미지 URL 수집 (개선된 버전)
    
    Args:
        driver: Selenium WebDriver 객체
        query: 검색어
        max_images: 수집할 최대 이미지 개수
    
    Returns:
        이미지 URL 리스트
    """
    image_urls = []
    
    try:
        # 구글 이미지 검색 URL 구성
        search_url = f"https://www.google.com/search?q={query}&tbm=isch"
        driver.get(search_url)
        
        # 페이지 로드 대기
        time.sleep(3)
        
        # 구글 이미지 썸네일 컨테이너 찾기 (더 정확한 셀렉터)
        # 구글 이미지 검색 결과의 구조에 맞는 셀렉터 사용
        thumbnail_selectors = [
            "div[data-ri] img",  # 구글 이미지 썸네일
            "img[data-src]",      # 지연 로딩 이미지
            "img[src]",           # 일반 이미지
            "div.rg_i img",       # 이미지 그리드 아이템
        ]
        
        image_elements = []
        for selector in thumbnail_selectors:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                if elements:
                    image_elements = elements
                    break
            except:
                continue
        
        # 스크롤하여 더 많은 이미지 로드
        for scroll in range(MAX_SCROLLS):
            # 페이지 끝까지 스크롤
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(SCROLL_PAUSE_TIME)
            
            # 새로운 이미지 요소 찾기
            for selector in thumbnail_selectors:
                try:
                    elements = driver.find_elements(By.CSS_SELECTOR, selector)
                    if len(elements) > len(image_elements):
                        image_elements = elements
                        break
                except:
                    continue
            
            if len(image_elements) >= max_images * 2:  # 여유있게 수집
                break
        
        # 이미지 URL 추출 (더 많은 시도)
        collected_count = 0
        for img in image_elements:
            if collected_count >= max_images * 2:  # 충분히 수집
                break
                
            try:
                # 여러 속성에서 URL 추출 시도
                img_url = None
                
                # 1. data-src (지연 로딩)
                img_url = img.get_attribute('data-src')
                
                # 2. src (직접 로딩)
                if not img_url:
                    img_url = img.get_attribute('src')
                
                # 3. data-imurl (구글 이미지 특수 속성)
                if not img_url:
                    img_url = img.get_attribute('data-imurl')
                
                if img_url:
                    # URL 정제
                    # 구글 썸네일 URL 처리
                    if 'googleusercontent.com' in img_url:
                        # 구글 이미지 썸네일 URL에서 원본 추출 시도
                        if '=s' in img_url or '=w' in img_url:
                            # 썸네일 크기 제한 제거 시도
                            if '=s' in img_url:
                                # =s0 (원본 크기)로 변경
                                img_url = img_url.split('=s')[0] + '=s0'
                            elif '=w' in img_url:
                                img_url = img_url.split('=w')[0] + '=w0-h0'
                    
                    # 유효한 HTTP URL인지 확인
                    if img_url.startswith('http') and img_url not in image_urls:
                        # 구글 로고나 아이콘 제외
                        if 'google.com/images' not in img_url.lower() or 'logo' not in img_url.lower():
                            image_urls.append(img_url)
                            collected_count += 1
                            
            except Exception as e:
                continue
        
        print(f"  검색어 '{query}': {len(image_urls)}개 이미지 URL 수집")
        
    except Exception as e:
        print(f"  경고: 검색 중 오류 발생 ({query}): {e}")
    
    return image_urls


# ================================================================================
# 4. 이미지 다운로드 및 저장
# ================================================================================
def download_image(url, output_path, timeout=10):
    """
    이미지 URL에서 이미지 다운로드 및 저장
    
    Args:
        url: 이미지 URL
        output_path: 저장할 파일 경로
        timeout: 요청 타임아웃 (초)
    
    Returns:
        성공 여부 (bool)
    """
    try:
        # HTTP 요청 헤더 설정
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        
        # 이미지 다운로드
        response = requests.get(url, headers=headers, timeout=timeout, stream=True)
        response.raise_for_status()
        
        # 이미지 데이터를 메모리에서 로드
        image_data = BytesIO(response.content)
        image = Image.open(image_data)
        
        # RGB로 변환 (RGBA, Grayscale 등 처리)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 이미지 저장
        image.save(output_path, 'JPEG', quality=95)
        
        return True
        
    except Exception as e:
        print(f"    다운로드 실패: {e}")
        return False


# ================================================================================
# 5. 이미지 해시 계산 (중복 제거용)
# ================================================================================
def calculate_image_hash(image_path):
    """이미지 파일의 해시값 계산"""
    try:
        with open(image_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except:
        return None


# ================================================================================
# 6. 메인 실행 코드
# ================================================================================
print("=" * 80)
print("구글 이미지에서 사람 얼굴 사진 다운로드")
print("=" * 80)

# 폴더 설정
setup_output_directory()

# 기존 이미지 해시 수집 (중복 제거용)
existing_hashes = set()
if os.path.exists(OUTPUT_DIR):
    for filename in os.listdir(OUTPUT_DIR):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            filepath = os.path.join(OUTPUT_DIR, filename)
            img_hash = calculate_image_hash(filepath)
            if img_hash:
                existing_hashes.add(img_hash)

print(f"\n기존 이미지: {len(existing_hashes)}개")
print(f"다운로드 목표: {DOWNLOAD_COUNT}개")

# WebDriver 설정
print("\n브라우저 초기화 중...")
driver = setup_selenium_driver()
print("브라우저 초기화 완료!")

# 이미지 URL 수집
print("\n" + "=" * 80)
print("이미지 URL 수집 중...")
print("=" * 80)

all_image_urls = []
collected_queries = set()

# 다양한 검색어로 이미지 URL 수집
max_attempts = len(SEARCH_QUERIES)  # 최대 검색어 개수만큼 시도
attempt_count = 0

while len(all_image_urls) < DOWNLOAD_COUNT * 3 and attempt_count < max_attempts:  # 더 여유있게 수집
    # 랜덤 검색어 선택
    query = random.choice(SEARCH_QUERIES)
    
    # 이미 사용한 검색어는 제외 (다양성 확보)
    if query in collected_queries and len(collected_queries) < len(SEARCH_QUERIES):
        attempt_count += 1
        continue
    
    collected_queries.add(query)
    attempt_count += 1
    print(f"\n검색어: '{query}' (시도 {attempt_count}/{max_attempts})")
    
    # 이미지 URL 수집
    urls = search_google_images(driver, query, max_images=30)  # 더 많이 수집
    all_image_urls.extend(urls)
    
    print(f"  현재까지 수집된 URL: {len(all_image_urls)}개")
    
    # 요청 간 딜레이
    time.sleep(DELAY_BETWEEN_REQUESTS)
    
    # 충분한 URL 수집 시 중단
    if len(all_image_urls) >= DOWNLOAD_COUNT * 3:
        break

# 중복 제거
all_image_urls = list(set(all_image_urls))
random.shuffle(all_image_urls)  # 랜덤 순서로 섞기

print(f"\n총 {len(all_image_urls)}개의 고유한 이미지 URL 수집 완료")

# 이미지 다운로드
print("\n" + "=" * 80)
print("이미지 다운로드 중...")
print("=" * 80)

downloaded_count = 0
failed_count = 0
duplicate_count = 0
skipped_count = 0  # 건너뛴 URL 개수

print(f"수집된 URL 개수: {len(all_image_urls)}개")
print(f"목표 다운로드 개수: {DOWNLOAD_COUNT}개\n")

for idx, url in enumerate(all_image_urls, 1):
    if downloaded_count >= DOWNLOAD_COUNT:
        break
    
    # URL 유효성 간단 체크
    if not url or len(url) < 10:
        skipped_count += 1
        continue
    
    print(f"\n[{idx}/{len(all_image_urls)}] 이미지 다운로드 시도...")
    print(f"  URL: {url[:80]}...")
    
    try:
        # 파일명 생성
        filename = f"image_{downloaded_count + 1:04d}.jpg"
        output_path = os.path.join(OUTPUT_DIR, filename)
        
        # 이미지 다운로드 (재시도 로직 포함)
        success = False
        for retry in range(2):  # 최대 2번 재시도
            if download_image(url, output_path, timeout=15):  # 타임아웃 증가
                success = True
                break
            if retry < 1:
                print(f"    재시도 중... ({retry + 1}/2)")
                time.sleep(1)
        
        if success:
            # 중복 체크
            img_hash = calculate_image_hash(output_path)
            
            if img_hash in existing_hashes:
                # 중복 이미지 삭제
                os.remove(output_path)
                duplicate_count += 1
                print(f"  중복 이미지 감지, 건너뜀 (총 {duplicate_count}개)")
            else:
                existing_hashes.add(img_hash)
                downloaded_count += 1
                print(f"  ✓ 다운로드 완료: {filename} ({downloaded_count}/{DOWNLOAD_COUNT})")
        else:
            failed_count += 1
            print(f"  ✗ 다운로드 실패 (총 {failed_count}개 실패)")
            
    except Exception as e:
        failed_count += 1
        print(f"  ✗ 오류 발생: {e} (총 {failed_count}개 실패)")
    
    # 요청 간 딜레이
    time.sleep(DELAY_BETWEEN_REQUESTS)
    
    # 진행 상황 출력 (10개마다)
    if idx % 10 == 0:
        print(f"\n[진행 상황] 성공: {downloaded_count}/{DOWNLOAD_COUNT}, 실패: {failed_count}, 중복: {duplicate_count}")

# WebDriver 종료
driver.quit()

# 결과 요약
print("\n" + "=" * 80)
print("다운로드 완료!")
print("=" * 80)
print(f"\n다운로드 성공: {downloaded_count}개")
print(f"다운로드 실패: {failed_count}개")
print(f"중복 제거: {duplicate_count}개")
print(f"건너뛴 URL: {skipped_count}개")
print(f"수집된 총 URL: {len(all_image_urls)}개")
print(f"\n저장 위치: {OUTPUT_DIR}/")

# 최종 이미지 개수 확인
final_images = [f for f in os.listdir(OUTPUT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
print(f"폴더 내 총 이미지: {len(final_images)}개")

print("\n" + "=" * 80)

