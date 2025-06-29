import matplotlib.pyplot as plt
from collections import Counter
import matplotlib.font_manager as fm
import os

# 한글 폰트 경로 (Windows 기준, 'malgun.ttf'가 일반적으로 설치되어 있음)
font_path = 'C:/Windows/Fonts/malgun.ttf'
if os.path.exists(font_path):
    plt.rc('font', family=fm.FontProperties(fname=font_path).get_name())
    plt.rcParams['axes.unicode_minus'] = False  # 마이너스 깨짐 방지
else:
    print("한글 폰트가 없어 기본 폰트로 출력됩니다.")

# train.txt 파일 읽기
with open('train.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 라벨 추출 (각 줄의 마지막 세미콜론 뒤가 라벨)
labels = [line.strip().split(';')[-1] for line in lines if ';' in line]

# 라벨 종류와 개수 세기
label_counts = Counter(labels)

print('라벨 종류:', list(label_counts.keys()))
print('라벨별 개수:', label_counts)

# 시각화
plt.figure(figsize=(8, 5))
plt.bar(list(label_counts.keys()), list(label_counts.values()), color='skyblue')
plt.title('감정 라벨별 데이터 개수')
plt.xlabel('감정 라벨')
plt.ylabel('개수')
plt.tight_layout()
plt.savefig('emotion_label_distribution.png')
plt.show() 