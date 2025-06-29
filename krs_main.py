import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import ssl
import matplotlib.font_manager as fm
import os # 파일 저장을 위해 os 모듈 추가

# --- Matplotlib 한글 폰트 설정 (Windows) ---
# 폰트가 설치되어 있는 경로를 확인하려면 제어판 -> 글꼴 로 이동합니다.
font_path = 'C:/Windows/Fonts/malgun.ttf' # 또는 'C:/Windows/Fonts/malgunbd.ttf' (볼드체)

# 폰트 파일 존재 여부 확인 (오류 방지)
if not os.path.exists(font_path):
    print(f"경고: 폰트 파일이 '{font_path}' 경로에 없습니다. 시스템의 다른 폰트 경로를 확인하거나 설치하세요.")
    # 기본 폰트로 대체하거나, 오류가 발생하지 않도록 처리
    # 예를 들어, font_path = 'C:/Windows/Fonts/arial.ttf' 등으로 대체할 수 있습니다.
    # 여기서는 진행을 위해 계속 진행하지만, 한글이 깨질 수 있습니다.
else:
    font_name = fm.FontProperties(fname=font_path).get_name()
    plt.rcParams['font.family'] = font_name
    plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지
    print(f"Matplotlib 한글 폰트 설정 완료: {font_name}")


# -----------------------------------------------------------------------------
# SSL 인증서 문제 해결 (선택 사항)
# -----------------------------------------------------------------------------
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")

# --- 1. 데이터 로드 및 탐색 ---
print("\n--- 1. 데이터 로드 및 탐색 ---")
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

class_names = ['비행기', '자동차', '새', '고양이', '사슴',
               '개', '개구리', '말', '배', '트럭'] # 한글 클래스 이름

print(f"훈련 데이터 이미지 개수: {x_train.shape[0]}, 이미지 형태: {x_train.shape[1:]}")
print(f"테스트 데이터 이미지 개수: {x_test.shape[0]}, 이미지 형태: {x_test.shape[1:]}")

# y_train 및 y_test의 고유 값과 형태 확인 (디버깅용)
print(f"y_train 고유 값: {np.unique(y_train)}")
print(f"y_test 고유 값: {np.unique(y_test)}")
print(f"y_train 형태: {y_train.shape}")
print(f"y_test 형태: {y_test.shape}")


# --- 이미지 샘플 시각화 ---
print("\n--- 이미지 샘플 시각화 ---")
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i])
    # y_train이 (N, 1) 형태이므로 y_train[i][0]으로 접근
    plt.xlabel(class_names[y_train[i][0]])
plt.suptitle("CIFAR-10 훈련 데이터 샘플 (25개)", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('cifar10_train_samples.png') # 이미지 파일로 저장
plt.show() # 창으로 표시


# --- 2. 데이터 전처리 ---
print("\n--- 2. 데이터 전처리 ---")
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
print(f"훈련 이미지 정규화 후 최소값: {x_train.min()}, 최대값: {x_train.max()}")


# --- 3. 모델 구축 (CNN 모델) ---
print("\n--- 3. 모델 구축 ---")
model = keras.Sequential([
    keras.Input(shape=(32, 32, 3)),

    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Flatten(),

    layers.Dense(256, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    layers.Dense(10, activation="softmax"), # 10개 클래스
])

model.summary()


# --- 4. 모델 컴파일 ---
print("\n--- 4. 모델 컴파일 ---")
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])


# --- 5. 모델 훈련 ---
print("\n--- 5. 모델 훈련 ---")
EPOCHS = 20
BATCH_SIZE = 64

history = model.fit(x_train, y_train,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_split=0.1)


# --- 6. 모델 평가 ---
print("\n--- 6. 모델 평가 ---")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\n테스트 정확도: {test_acc:.4f}")
print(f"테스트 손실: {test_loss:.4f}")


# --- 7. 훈련 결과 시각화 ---
print("\n--- 7. 훈련 결과 시각화 ---")
plt.figure(figsize=(12, 5))

# 정확도 그래프
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='훈련 정확도')
plt.plot(history.history['val_accuracy'], label='검증 정확도')
plt.title('정확도 변화')
plt.xlabel('에포크')
plt.ylabel('정확도')
plt.legend()
plt.grid(True)

# 손실 그래프
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='훈련 손실')
plt.plot(history.history['val_loss'], label='검증 손실')
plt.title('손실 변화')
plt.xlabel('에포크')
plt.ylabel('손실')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('cifar10_training_results.png') # 이미지 파일로 저장
plt.show()


# --- 8. 예측 수행 및 결과 확인 ---
print("\n--- 8. 예측 수행 및 결과 확인 ---")
# 테스트 데이터셋 전체에 대해 예측 수행 (일부만 하면 predicted_labels의 크기가 작아질 수 있음)
# 오류 발생 시점 (line 179)을 고려하여 x_test 전체에 대해 예측해보고 디버깅하는 것이 좋습니다.
# 아니면 문제 발생 이전에 테스트 데이터의 레이블 범위를 확인하는 것이 중요합니다.
predictions = model.predict(x_test) # x_test[:10] 대신 x_test 전체로 변경 (예시)
predicted_labels = np.argmax(predictions, axis=1)

print(f"실제 레이블 (첫 10개): {y_test[:10].flatten()}")
print(f"예측된 레이블 (첫 10개): {predicted_labels[:10]}") # 첫 10개만 출력

# 디버깅을 위해 모든 예측된 레이블의 고유 값과 최대값 확인
print(f"모든 예측된 레이블의 고유 값: {np.unique(predicted_labels)}")
print(f"모든 예측된 레이블의 최대값: {np.max(predicted_labels)}")


print("클래스 이름으로 변환:")
# 이 for 루프가 문제의 179번째 줄 근처일 가능성이 높습니다.
# 오류가 발생한 위치를 명확히 알기 위해 해당 줄의 `i` 값과 `predicted_labels[i]` 값을 출력해 보세요.
for i in range(10): # y_test[:10]과 predicted_labels[:10]의 길이에 맞춤
    try:
        true_label_name = class_names[y_test[i][0]]
        predicted_label_name = class_names[predicted_labels[i]]
        print(f"실제: {true_label_name}, 예측: {predicted_label_name}")
    except IndexError as e:
        print(f"오류 발생! i = {i}, y_test[{i}][0] = {y_test[i][0]}, predicted_labels[{i}] = {predicted_labels[i]}")
        print(f"오류 내용: {e}")
        break # 오류 발생 시 루프 중단

# --- 9. 일부 예측 결과 시각화 ---
print("\n--- 9. 일부 예측 결과 시각화 ---")
plt.figure(figsize=(12, 12))
# range(25)는 고정된 값입니다. predicted_labels에 10이 있다면 이 시각화에서도 문제가 발생합니다.
# 안전하게 시각화하려면, predicted_labels에서 10을 포함하지 않는 샘플만 선택하거나
# 시각화할 이미지 수를 예측 결과의 실제 길이로 제한해야 합니다.
# 여기서는 오류 디버깅을 위해 일단 25개 시각화를 유지합니다.
for i in range(25):
    if i >= len(x_test) or i >= len(predicted_labels): # 리스트 범위를 벗어나지 않도록 방어 코드
        break
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_test[i])
    
    # 여기서 IndexError가 발생할 가능성이 높습니다.
    # predicted_labels[i]가 10을 포함할 경우.
    try:
        true_label = class_names[y_test[i][0]]
        predicted_label = class_names[predicted_labels[i]]
        color = 'blue' if predicted_label == true_label else 'red'
        plt.xlabel(f"예측: {predicted_label}\n실제: {true_label}", color=color)
    except IndexError as e:
        plt.xlabel(f"예측 오류! (인덱스 {predicted_labels[i]})", color='red')
        print(f"시각화 중 오류 발생: i = {i}, predicted_labels[{i}] = {predicted_labels[i]}, 오류: {e}")

plt.suptitle("CIFAR-10 테스트 데이터 예측 결과 (25개)", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('cifar10_prediction_results.png') # 이미지 파일로 저장
plt.show()