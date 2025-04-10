import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from collections import Counter

# ============================
# 설정
# ============================
VIDEO_PATH = "C:/Users/goodf/Desktop/test3.mp4"
CONFIDENCE_THRESHOLD = 0.5  # 신뢰도 기준값

# ============================
# 모델 & 인코더 로드
# ============================
model = load_model('C:/Users/goodf/Desktop/project/sign_model_u128_l2_d25.h5')
encoder = joblib.load('label_encoder.pkl')

# ============================
# MediaPipe 초기화
# ============================
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
pose = mp_pose.Pose()
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)

# ============================
# 영상 로드
# ============================
cap = cv2.VideoCapture(VIDEO_PATH)
sequence = []
predictions = []

print(f"[INFO] 영상 시작: {VIDEO_PATH}")

# ============================
# 프레임 처리 루프
# ============================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    pose_result = pose.process(image)
    hands_result = hands.process(image)

    keypoints = []

    # 포즈 키포인트 (어깨~손목)
    if pose_result.pose_landmarks:
        for idx in [11, 13, 15, 12, 14, 16]:
            lm = pose_result.pose_landmarks.landmark[idx]
            keypoints.extend([lm.x, lm.y, lm.z])
    else:
        keypoints.extend([0] * 18)

    # 손 키포인트
    if hands_result.multi_hand_landmarks:
        hands_data = hands_result.multi_hand_landmarks[:2]
        for hand in hands_data:
            for lm in hand.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
        if len(hands_data) == 1:
            keypoints.extend([0] * 63)
    else:
        keypoints.extend([0] * 126)

    if len(keypoints) != 144:
        continue

    sequence.append(keypoints)

    if len(sequence) == 30:
        input_data = np.expand_dims(sequence, axis=0)
        prediction = model.predict(input_data)[0]
        confidence = np.max(prediction)
        pred_label = encoder.inverse_transform([np.argmax(prediction)])[0]

        # 🔽 디버깅용 무조건 출력
        print(f"[DEBUG] 예측: {pred_label} | confidence: {confidence:.2f}")

        if confidence > CONFIDENCE_THRESHOLD:
            predictions.append(pred_label)
            print(f'🟢 예측 결과: {pred_label} ({confidence:.2f})')
        else:
            print(f'🔹 무시됨: {pred_label} ({confidence:.2f})')

        sequence = []

# ============================
# 결과 출력
# ============================
cap.release()
cv2.destroyAllWindows()

if predictions:
    print(f"\n전체 예측 흐름: {' → '.join(predictions)}")
    final_word = Counter(predictions).most_common(1)[0][0]
    print(f"최종 번역 결과: {final_word}")
else:
    print("\n예측된 결과가 없습니다.")
