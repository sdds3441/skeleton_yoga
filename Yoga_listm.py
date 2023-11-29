import cv2
from cvzone.HandTrackingModule import HandDetector
import socket
import numpy as np
from keras.models import load_model
import mediapipe as mp
import timeit

width, height = 1280, 720
# Webcam
actions = ['Bhujasana', 'Padamasana', 'Tadasana']
seq_length = 30

cap = cv2.VideoCapture(
    "C://Users//김효찬//Desktop//수업//3학년//2학기//AI프로그래밍//archive//dataset//test//Bhujasana//video 13.mp4")
cap.set(3, width)
cap.set(4, height)

model = load_model('models/model.h5')

# Hand Detect
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

buttonDelay = 10
buttonPressed = False
buttonCounter = 0
addObject = '0'

seq = []
action_seq = []
action_list = []
time_counter = True
first = True

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        success, frame = cap.read()
        #if first:
        cv2.waitKey(100)
          #  first = False

        start_t = timeit.default_timer()

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = holistic.process(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )

        if results.pose_landmarks.landmark is not None:

            for i, lm in enumerate(results.pose_landmarks.landmark):

                if lm.visibility < 0.01:
                    visible = False
                else:
                    visible = True

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

            if len(seq) <= seq_length:
                continue

            cv2.imshow('22', image)
            print('각도계산')
            input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)

            y_pred = model.predict(input_data).squeeze()
            ("모델계산완료")
            i_pred = int(np.argmax(y_pred))
            conf = y_pred[i_pred]

            if conf < 0.9:
                continue

            action = actions[i_pred]
            action_seq.append(action)

            if len(action_seq) < 3:
                continue

            action_list = []
            print(action)

            if action_seq[-1] == action_seq[-2] == action_seq[-3]:
                action_list.append(action)

            else:
                action_list = ["None"]

        img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
        cv2.imshow("Image", img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
