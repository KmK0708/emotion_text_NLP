import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
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

print("=== 감정 분석 예측 테스트 시작 ===")

# 1. 모델 및 전처리 도구 로드
print("1. 모델 로드 중...")
model = load_model('emotion_model_improved.h5')
print("모델 로드 완료!")

# 2. 토크나이저와 라벨 인코더 재생성 (훈련 시와 동일하게)
print("\n2. 전처리 도구 설정 중...")

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

# 훈련 데이터로 토크나이저 재생성
train_texts, train_labels = load_data('train.txt')
tokenizer = Tokenizer(num_words=15000, oov_token="<OOV>")
tokenizer.fit_on_texts(train_texts)

# 라벨 인코더 재생성
label_encoder = LabelEncoder()
all_labels = train_labels
label_encoder.fit(all_labels)

print("라벨 매핑:")
for i, label in enumerate(label_encoder.classes_):
    print(f"  {i}: {label}")

# 3. 예측 함수 정의
def predict_emotion(text, model, tokenizer, label_encoder, max_length=100):
    """텍스트의 감정을 예측하는 함수"""
    # 텍스트 전처리
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')
    
    # 예측
    prediction = model.predict(padded, verbose=0)
    predicted_label = np.argmax(prediction[0])
    confidence = np.max(prediction[0])
    
    # 결과
    emotion = label_encoder.classes_[predicted_label]
    
    return emotion, confidence, prediction[0]

# 4. 미리 정의된 테스트 문장들
print("\n3. 미리 정의된 테스트 문장들로 예측...")
test_sentences = [
    "I am feeling so happy today!",
    "I am really sad and depressed",
    "I am angry about what happened",
    "I am scared of the dark",
    "I love you so much",
    "I am surprised by the news",
    "I feel excited about the trip",
    "I am worried about the future",
    "I am grateful for everything",
    "I am frustrated with this situation"
]

print("예측 결과:")
print("-" * 80)
for i, sentence in enumerate(test_sentences, 1):
    emotion, confidence, all_probs = predict_emotion(sentence, model, tokenizer, label_encoder)
    print(f"{i:2d}. '{sentence}'")
    print(f"    예측 감정: {emotion:10} | 신뢰도: {confidence:.3f}")
    print()

# 5. 사용자 입력 테스트
print("\n4. 사용자 입력 테스트...")
print("영어로 감정이 담긴 문장을 입력하세요 (종료하려면 'quit' 입력):")

while True:
    user_input = input("\n텍스트 입력: ").strip()
    
    if user_input.lower() == 'quit':
        break
    
    if not user_input:
        print("텍스트를 입력해주세요.")
        continue
    
    # 예측 수행
    emotion, confidence, all_probs = predict_emotion(user_input, model, tokenizer, label_encoder)
    
    print(f"\n예측 결과:")
    print(f"  입력 텍스트: {user_input}")
    print(f"  예측 감정: {emotion}")
    print(f"  신뢰도: {confidence:.3f}")
    
    # 모든 감정의 확률 출력
    print(f"\n모든 감정 확률:")
    for i, (label, prob) in enumerate(zip(label_encoder.classes_, all_probs)):
        print(f"  {label:10}: {prob:.3f}")

print("\n=== 예측 테스트 완료! ===")

# 6. 시각화 예시 (마지막 예측 결과)
if 'user_input' in locals() and user_input.lower() != 'quit':
    print("\n5. 마지막 예측 결과 시각화...")
    
    emotions = label_encoder.classes_
    probabilities = all_probs
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(emotions, probabilities, color=['red', 'orange', 'yellow', 'green', 'blue', 'purple'])
    plt.title(f'감정 분석 결과: "{user_input}"')
    plt.xlabel('감정')
    plt.ylabel('확률')
    plt.ylim(0, 1)
    
    # 막대 위에 확률 값 표시
    for bar, prob in zip(bars, probabilities):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{prob:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('emotion_prediction_result.png')
    plt.show() 