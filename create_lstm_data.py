import os
import time
import cv2
import mediapipe as mp
import numpy as np

actions = ['Bhujasana', 'Padamasana', 'Tadasana', 'Trikasana', 'Vrikshasana']
seq_length = 10
secs_for_action = 10

# MediaPipe hands model
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(
    "C://Users//김효찬//Desktop//수업//3학년//2학기//AI프로그래밍//archive//dataset//highlight//Bhujasana//video 1.mp4")

created_time = int(time.time())
#os.makedis('../dataset', exist_ok=True)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for idx, action in enumerate(actions):
        print(action)
        cv2.waitKey(3000)
        while cap.isOpened():
            data = []
            ret, image = cap.read()

            image = cv2.flip(image, 1)

            cv2.putText(image, f'Waiting for collecting {action.upper()} action...', org=(10, 30),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

            cv2.imshow('img', image)
            cv2.waitKey(3000)


            start_time = time.time()

            while time.time() - start_time < secs_for_action:
                ret, image = cap.read()

                image = cv2.flip(image, 1)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.pose_landmarks is not None:
                    joint = np.zeros((33, 4))
                    for j, lm in enumerate(results.pose_landmarks.landmark):
                        joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                    # Compute angles between joints
                    v1 = joint[[11, 11, 13, 12, 14, 11, 12, 23, 23, 25, 24, 26], :3]  # Parent joint
                    v2 = joint[[12, 13, 15, 14, 16, 23, 24, 24, 25, 27, 26, 28], :3]  # Child joint
                    v = v2 - v1  # [20, 3]
                    # Normalize v
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                    # Get angle using arcos of dot product
                    angle = np.arccos(np.einsum('nt,nt->n',
                                                v[[0, 1, 0, 3, 0, 0, 5, 7, 8, 7, 10], :],
                                                v[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], :]))  # [15,]

                    angle = np.degrees(angle)  # Convert radian to degree

                    angle_label = np.array([angle], dtype=np.float32)
                    angle_label = np.append(angle_label, idx)

                    d = np.concatenate([joint.flatten(), angle_label])

                    data.append(d)

                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2,
                                                                     circle_radius=4),
                                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2,
                                                                     circle_radius=2)
                                              )
                cv2.imshow('img', image)
                if cv2.waitKey(1) == ord('q'):
                    break
            data = np.array(data)
            print(action, data.shape)
            np.save(os.path.join('dataset//raw', f'raw_{action}'), data)

            # Create sequence data
            full_seq_data = []
            for seq in range(len(data) - seq_length):
                full_seq_data.append(data[seq:seq + seq_length])

            full_seq_data = np.array(full_seq_data)
            print(action, full_seq_data.shape)
            np.save(os.path.join('dataset//seq', f'seq_{action}'), full_seq_data)

            break

        if action == 'Bhujasana':
            cap = cv2.VideoCapture(
                "C://Users//김효찬//Desktop//수업//3학년//2학기//AI프로그래밍//archive//dataset//highlight//Padamasana//video 1.mp4")
        elif action == 'Padamasana':
            cap = cv2.VideoCapture(
                "C://Users//김효찬//Desktop//수업//3학년//2학기//AI프로그래밍//archive//dataset//highlight//Tadasana//video 1.mp4")
     #   elif action == 'Tadasana':
      #      cap = cv2.VideoCapture(
      #          "C://Users//김효찬//Desktop//수업//3학년//2학기//AI프로그래밍//archive//dataset//highlight//Trikasana//video 1.mp4")
      #  elif action == 'Trikasana':
       #     cap = cv2.VideoCapture(
        #        "C://Users//김효찬//Desktop//수업//3학년//2학기//AI프로그래밍//archive//dataset//highlight//Vrikshasana//video 1.mp4")
