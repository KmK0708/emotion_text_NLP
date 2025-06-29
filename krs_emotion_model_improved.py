import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
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

print("=== 개선된 감정 분석 모델 구축 및 훈련 시작 ===")

# 1. 데이터 로드 및 전처리
def load_data(filename):
    texts = []
    labels = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            if ';' in line:
                text, label = line.strip().split(';', 1)
                texts.append(text)
                labels.append(label)
    return texts, labels

print("1. 데이터 로드 및 전처리 중...")
train_texts, train_labels = load_data('train.txt')
val_texts, val_labels = load_data('val.txt')
test_texts, test_labels = load_data('test.txt')

# 라벨 인코딩
label_encoder = LabelEncoder()
all_labels = train_labels + val_labels + test_labels
label_encoder.fit(all_labels)

train_labels_encoded = label_encoder.transform(train_labels)
val_labels_encoded = label_encoder.transform(val_labels)
test_labels_encoded = label_encoder.transform(test_labels)

# 텍스트 토큰화 (어휘 크기 증가)
tokenizer = Tokenizer(num_words=15000, oov_token="<OOV>")
tokenizer.fit_on_texts(train_texts)

train_sequences = tokenizer.texts_to_sequences(train_texts)
val_sequences = tokenizer.texts_to_sequences(val_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)

# 시퀀스 패딩
max_length = 100
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding='post', truncating='post')
val_padded = pad_sequences(val_sequences, maxlen=max_length, padding='post', truncating='post')
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding='post', truncating='post')

print(f"데이터 준비 완료:")
print(f"  훈련: {train_padded.shape}")
print(f"  검증: {val_padded.shape}")
print(f"  테스트: {test_padded.shape}")

# 2. 개선된 모델 구축
print("\n2. 개선된 모델 구축 중...")
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 256  # 임베딩 차원 증가
num_classes = len(label_encoder.classes_)

model = Sequential([
    Embedding(vocab_size, embedding_dim),
    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.2),
    Bidirectional(LSTM(64)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(num_classes, activation='softmax')
])

model.summary()

# 3. 모델 컴파일 (학습률 조정)
print("\n3. 모델 컴파일 중...")
model.compile(
    optimizer=Adam(learning_rate=0.001),  # 명시적 학습률 설정
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 4. 콜백 설정 (개선)
early_stopping = EarlyStopping(
    monitor='val_accuracy',  # 정확도 기준으로 변경
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

# 5. 모델 훈련 (더 많은 에포크)
print("\n4. 모델 훈련 시작...")
history = model.fit(
    train_padded,
    train_labels_encoded,
    epochs=50,  # 에포크 증가
    batch_size=64,  # 배치 크기 증가
    validation_data=(val_padded, val_labels_encoded),
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# 6. 모델 평가
print("\n5. 모델 평가 중...")
test_loss, test_accuracy = model.evaluate(test_padded, test_labels_encoded, verbose=0)
print(f"테스트 정확도: {test_accuracy:.4f}")
print(f"테스트 손실: {test_loss:.4f}")

# 7. 훈련 과정 시각화
print("\n6. 훈련 과정 시각화...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# 정확도 그래프
ax1.plot(history.history['accuracy'], label='훈련 정확도')
ax1.plot(history.history['val_accuracy'], label='검증 정확도')
ax1.set_title('모델 정확도')
ax1.set_xlabel('에포크')
ax1.set_ylabel('정확도')
ax1.legend()
ax1.grid(True)

# 손실 그래프
ax2.plot(history.history['loss'], label='훈련 손실')
ax2.plot(history.history['val_loss'], label='검증 손실')
ax2.set_title('모델 손실')
ax2.set_xlabel('에포크')
ax2.set_ylabel('손실')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('emotion_model_improved_training.png')
plt.show()

# 8. 예측 및 결과 확인
print("\n7. 예측 결과 확인...")
predictions = model.predict(test_padded[:10])
predicted_labels = np.argmax(predictions, axis=1)

print("예측 결과 (첫 10개):")
for i in range(10):
    true_label = label_encoder.classes_[test_labels_encoded[i]]
    pred_label = label_encoder.classes_[predicted_labels[i]]
    confidence = np.max(predictions[i])
    print(f"  실제: {true_label:10} | 예측: {pred_label:10} | 신뢰도: {confidence:.3f}")

# 9. 모델 저장
print("\n8. 모델 저장 중...")
model.save('emotion_model_improved.h5')
print("개선된 모델이 'emotion_model_improved.h5'로 저장되었습니다.")

print("\n=== 개선된 모델 훈련 완료! ===")
print(f"최종 테스트 정확도: {test_accuracy:.4f}")
print("이전 모델 대비 개선 사항:")
print("- Bidirectional LSTM 사용")
print("- 더 큰 임베딩 차원 (256)")
print("- 적절한 학습률 설정")
print("- 학습률 감소 스케줄러 추가")
print("- 더 많은 에포크 (50)")
print("- 정확도 기반 early stopping") 