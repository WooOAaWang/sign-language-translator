import os
import numpy as np
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dropout
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# 데이터 및 라벨 로드
###################### 라벨링으로 저장방법 달라서 여기 바뀜
def load_data_nested(data_path):
    sequences = []
    sequence_labels = []

    # 단어별 폴더 탐색 (못생기다, 오해, ...)
    for label in os.listdir(data_path):
        label_path = os.path.join(data_path, label)
        if not os.path.isdir(label_path):  # 안전 체크
            continue

        # 영상별 폴더 탐색 (video1, video2, ...)
        for video_folder in os.listdir(label_path):
            video_path = os.path.join(label_path, video_folder)
            if not os.path.isdir(video_path):
                continue

            # 시퀀스(.npy) 파일 읽기
            for file in os.listdir(video_path):
                if file.endswith('.npy'):
                    file_path = os.path.join(video_path, file)
                    sequence = np.load(file_path)
                    if sequence.shape == (30, 144):
                        sequences.append(sequence)
                        sequence_labels.append(label)

    return np.array(sequences), sequence_labels


# 모델 정의 및 컴파일
def build_model(input_shape, num_classes, num_units, num_layers=2, dropout_rate=0.5):
    model = Sequential()
    # 첫 번째 LSTM 층
    model.add(LSTM(num_units, return_sequences=True if num_layers > 1 else False, input_shape=input_shape))
    # 추가 LSTM 층
    for i in range(1, num_layers):
        model.add(LSTM(num_units, return_sequences=True if i < num_layers - 1 else False))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate)) ######################## Dropout추가해서 신경망 과적합 방지 / 무작위 선택된 뉴런 꺼서 일반화 올리고 공동적응 방지 목적
    # Dense 층
    model.add(Dense(num_units // 2, activation='relu'))
    # 출력 층
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# 학습 및 평가
####################################  EarlyStopping추가 : 30번 반복학습해서 손실 줄다가 다시 증가하면 멈춰주는 기능 / 오버피팅 낮춰주고 학습시간 짧아지고 성능 안정적유지긍
def train_and_evaluate(X_train, y_train, X_test, y_test, model):
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test), callbacks=[early_stopping])
    
    # 예측 및 성능 평가
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    ################################ 정확도 외에 평가 점수 추가함
    accuracy = accuracy_score(y_true_classes, y_pred_classes)
    precision = precision_score(y_true_classes, y_pred_classes, average='macro')
    recall = recall_score(y_true_classes, y_pred_classes, average='macro')
    f1 = f1_score(y_true_classes, y_pred_classes, average='macro')
    
    return history, accuracy, precision, recall, f1

# 메인함수
def main():
    DATA_PATH = 'sign_data'  # 데이터가 저장된 디렉토리의 경로를 정의
    X, y_labels = load_data_nested(DATA_PATH)    # 데이터를 불러오는 함수
    le = LabelEncoder()  # 라벨 인코더 생성
    y = le.fit_transform(y_labels)  # 문자열 라벨을 정수로 변환
    y_cat = to_categorical(y)  # 정수 라벨을 원-핫 인코딩


    joblib.dump(le, 'label_encoder.pkl')
    print(" 라벨 인코더 저장 완료 → label_encoder.pkl")
    
    ###################### stratify 옵션 클래스 비율 유지하면서 8:2 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=0.2, random_state=42, stratify=y)
    
    num_classes = len(np.unique(y_labels))  # 클래스 수 계산


    #################### 실험할 파라미터 수랑  LSTM 층의 수와 드롭아웃 비율을 설정
    unit_options = [64, 128]   
    num_layers_options = [1, 2]  # LSTM 층의 수를 다양하게 설정 1,2,3
    dropout_options = [0, 0.25]  # 드롭아웃 비율을 다양하게 설정 0, 0.25, 0.5
    

    # 각 조합에 대해 모델을 훈련하고 평가
    for num_units in unit_options:
        for num_layers in num_layers_options:
            for dropout_rate in dropout_options:
                print(f"\n Training with {num_units} units, {num_layers} layers, dropout {dropout_rate}")
                model = build_model((30, 144), num_classes, num_units, num_layers, dropout_rate)
                history, accuracy, precision, recall, f1 = train_and_evaluate(X_train, y_train, X_test, y_test, model)

                print(f" Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")
                # 출력은 각 파리미터 수 별 각 LSTM층 별 각 dropout비율별 정확도, 정밀도, 재현율, F1 이 출력됨됨

                # u(파라미터수)_l(층수)_d(드롭아웃비율)로 저장됨됨
                model_name = f"sign_model_u{num_units}_l{num_layers}_d{int(dropout_rate*100)}.h5"
                model.save(model_name)
                print(f" 모델 저장 완료 → {model_name}")



if __name__ == "__main__":
    main()
