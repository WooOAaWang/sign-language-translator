import os
import cv2
import mediapipe as mp
import numpy as np

# ========== 사용자 설정 ==========
root_video_dir = "C:/Users/goodf/Desktop/project/SLV/못생기다"  # ❗ 실제 영상 폴더 경로
label = os.path.basename(root_video_dir)  # 라벨 이름 = 폴더명
save_root = os.path.join('sign_data', label)  # 저장 디렉토리



# MediaPipe 초기화
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
pose = mp_pose.Pose()
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)

# 영상 목록 불러오기
video_files = [f for f in os.listdir(root_video_dir) if f.endswith('.mp4')]
print(f"[INFO] 총 영상 파일 수: {len(video_files)}개")

for idx, video_file in enumerate(video_files):
    video_path = os.path.join(root_video_dir, video_file)
    video_name = f"video{idx + 1}"
    save_dir = os.path.join(save_root, video_name)
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\n🎥 영상 처리 시작: {video_file} → {video_name} (총 프레임 수: {total_frames})")

    sequence = []
    saved_count = 0
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            print("⚠️ 프레임 없음 — 영상 끝 또는 실패")
            break

        frame_idx += 1
        print(f"📸 현재 프레임: {frame_idx}/{total_frames}", end=' | ')

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if image is None or image.shape[0] == 0 or image.shape[1] == 0:
            print("❌ 이미지 shape 오류")
            continue

        image.flags.writeable = False
        pose_result = pose.process(image)
        hands_result = hands.process(image)

        keypoints = []

        # 포즈
        if pose_result.pose_landmarks:
            for idx in [11, 13, 15, 12, 14, 16]:
                lm = pose_result.pose_landmarks.landmark[idx]
                keypoints.extend([lm.x, lm.y, lm.z])
            print("👤 포즈 인식됨", end=' | ')
        else:
            keypoints.extend([0] * 18)
            print("👤 포즈 없음", end=' | ')

        # 손
        if hands_result.multi_hand_landmarks:
            print(f"🖐️ 손 인식: {len(hands_result.multi_hand_landmarks)}개", end=' | ')
            detected_hands = hands_result.multi_hand_landmarks[:2]
            for hand in detected_hands:
                for lm in hand.landmark:
                    keypoints.extend([lm.x, lm.y, lm.z])
            if len(detected_hands) == 1:
                keypoints.extend([0] * 63)
        else:
            keypoints.extend([0] * 126)
            print("🖐️ 손 인식 안됨", end=' | ')

        if len(keypoints) != 144:
            print("⚠️ 키포인트 개수 이상 → 건너뜀")
            continue

        sequence.append(keypoints)
        print(f"🧩 시퀀스 길이: {len(sequence)}/30")

        if len(sequence) == 30:
            np.save(os.path.join(save_dir, f"{saved_count}.npy"), np.array(sequence))
            print(f"✅ 저장됨: {saved_count}.npy")
            sequence = []
            saved_count += 1

    cap.release()
    print(f"\n🧾 {video_name} 저장 완료 | 총 시퀀스: {saved_count}")

cv2.destroyAllWindows()
print("\n🎉 전체 영상 처리 완료")
