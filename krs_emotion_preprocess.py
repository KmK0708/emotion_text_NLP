import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# 한글 폰트 설정
font_path = 'C:/Windows/Fonts/malgun.ttf'
if os.path.exists(font_path):
    plt.rc('font', family=fm.FontProperties(fname=font_path).get_name())
    plt.rcParams['axes.unicode_minus'] = False
else:
    print("한글 폰트가 없어 기본 폰트로 출력됩니다.")

print("=== 감정 분석 데이터 전처리 시작 ===")

# 1. 데이터 로드 및 분리
def load_data(filename):
    """데이터 파일을 읽어서 텍스트와 라벨을 분리"""
    texts = []
    labels = []
    
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            if ';' in line:
                text, label = line.strip().split(';', 1)
                texts.append(text)
                labels.append(label)
    
    return texts, labels

print("1. 데이터 로드 중...")
train_texts, train_labels = load_data('train.txt')
val_texts, val_labels = load_data('val.txt')
test_texts, test_labels = load_data('test.txt')

print(f"훈련 데이터: {len(train_texts)}개")
print(f"검증 데이터: {len(val_texts)}개")
print(f"테스트 데이터: {len(test_texts)}개")

# 2. 라벨 인코딩 (문자열 -> 숫자)
print("\n2. 라벨 인코딩 중...")
label_encoder = LabelEncoder()
# 전체 데이터셋의 모든 라벨을 하나로 합치기
# 이렇게 하는 이유: 모든 가능한 라벨 종류를 파악하기 위해
all_labels = train_labels + val_labels + test_labels
# 전체 라벨로 인코더 학습 (fit)
# 이 단계에서 모든 고유한 라벨을 찾고 알파벳 순으로 정렬하여 숫자 할당
label_encoder.fit(all_labels)

# 각 데이터셋별로 라벨을 숫자로 변환 (transform)
train_labels_encoded = label_encoder.transform(train_labels)
val_labels_encoded = label_encoder.transform(val_labels)
test_labels_encoded = label_encoder.transform(test_labels)
# 라벨 매핑 결과 출력 (어떤 문자열이 어떤 숫자로 매핑되었는지 확인)
print("라벨 매핑:")
for i, label in enumerate(label_encoder.classes_):
    print(f"  {i}: {label}")

# 3. 텍스트 토큰화
print("\n3. 텍스트 토큰화 중...")
# Tokenizer 객체 생성
# num_words=10000: 가장 빈번한 상위 10,000개 단어만 사용
# oov_token="<OOV>": 사전에 없는 단어를 처리하는 토큰
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
# 훈련 데이터로만 토크나이저 학습 (fit)
tokenizer.fit_on_texts(train_texts)
# 텍스트를 시퀀스로 변환
# 각 텍스트를 토큰 시퀀스로 변환
train_sequences = tokenizer.texts_to_sequences(train_texts)
val_sequences = tokenizer.texts_to_sequences(val_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)
# 어휘 크기: 토큰화된 단어의 총 개수
# 예: 10000개의 단어가 있으면 10001개의 인덱스 사용
# +1은 특별 토큰을 위한 공간
print(f"어휘 크기: {len(tokenizer.word_index) + 1}")
# 토크나이저 사용 예시
# 토큰화된 문장을 숫자 시퀀스로 변환
sequences = tokenizer.texts_to_sequences(["hello, today is a good day. is it?"])
print(sequences)
# 토큰화된 문장을 패딩 처리
padded_sequences = pad_sequences(sequences, maxlen=10, padding='post', truncating='post')
print(padded_sequences)

# 4. 시퀀스 패딩 (길이 맞추기)
print("\n4. 시퀀스 패딩 중...")
max_length = 100  # 최대 길이 설정
# padding='post': 문장 뒤쪽에 0을 채워넣기 (앞쪽은 'pre')
# truncating='post': 100보다 긴 문장은 뒷부분을 잘라내기 (앞쪽은 'pre')
# 훈련 데이터에만 적용
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding='post', truncating='post')
val_padded = pad_sequences(val_sequences, maxlen=max_length, padding='post', truncating='post')
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding='post', truncating='post')

print(f"패딩 후 형태:")
print(f"  훈련 데이터: {train_padded.shape}")
print(f"  검증 데이터: {val_padded.shape}")
print(f"  테스트 데이터: {test_padded.shape}")

# 5. 데이터 분포 시각화
print("\n5. 데이터 분포 시각화...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# 라벨 분포
label_counts = pd.Series(train_labels).value_counts()
ax1.bar(label_counts.index, label_counts.values, color='skyblue')
ax1.set_title('감정 라벨 분포')
ax1.set_xlabel('감정')
ax1.set_ylabel('개수')
ax1.tick_params(axis='x', rotation=45)

# 텍스트 길이 분포
text_lengths = [len(text.split()) for text in train_texts]
ax2.hist(text_lengths, bins=50, color='lightgreen', alpha=0.7)
ax2.set_title('텍스트 길이 분포')
ax2.set_xlabel('단어 개수')
ax2.set_ylabel('빈도')
ax2.axvline(np.mean(text_lengths), color='red', linestyle='--', label=f'평균: {np.mean(text_lengths):.1f}')
ax2.legend()

plt.tight_layout()
plt.savefig('emotion_data_analysis.png')
plt.show()

# 6. 샘플 데이터 확인
print("\n6. 샘플 데이터 확인:")
print("원본 텍스트:", train_texts[0])
print("토큰화:", train_sequences[0][:10], "...")
print("패딩 후:", train_padded[0][:10], "...")
print("라벨:", train_labels[0], "->", train_labels_encoded[0])

print("\n=== 전처리 완료! ===")
print("다음 단계에서 사용할 데이터:")
print(f"  X_train: {train_padded.shape}")
print(f"  y_train: {train_labels_encoded.shape}")
print(f"  X_val: {val_padded.shape}")
print(f"  y_val: {val_labels_encoded.shape}")
print(f"  X_test: {test_padded.shape}")
print(f"  y_test: {test_labels_encoded.shape}") 