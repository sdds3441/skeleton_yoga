import timeit
import mediapipe as mp
import cv2
import numpy as np
import threading
import sys


def cal_extend_data(pose_landmarks):
    h_joint = np.zeros((33, 3))
    for j, lm in enumerate(pose_landmarks.landmark):
        x = lm.x
        y = lm.y
        z = lm.z

        data.extend([round(x * width), round(height - (y * height)), round(z, 5)])
        h_joint[j] = [lm.x, lm.y, lm.z]
    return h_joint


def cal_gesture(hand_joint):
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

    joint_data = np.array([angle], dtype=np.float32)

    return joint_data


def timer():
    global sec
    sec += 1

    timers = threading.Timer(1, timer)
    timers.start()

   # if sec == 3:
      #  timers.cancel()


rps_gesture = {0: 'Bhujasana', 1: 'Padamasana', 2: 'Tadasana', 3: 'Trikasana', 4: 'Vrikshasana'}

width = 1280
height = 720

cap = cv2.VideoCapture("C://Users//김효찬//Desktop//수업//3학년//2학기//인공지능프로그래밍//yoga//highlight//Trikasana//video 3.mp4")
cap.set(3, width)
cap.set(4, height)

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

sec = 0
pos_idx = 6
first = True
counter = 0

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for idx, action in enumerate(rps_gesture):
        while cap.isOpened():
            ret, frame = cap.read()

            start_t = timeit.default_timer()

            data = []
            R_action = 0
            L_action = 0

            # Recolor Feed
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Make Detections
            results = holistic.process(image)

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )

            if results.pose_landmarks is not None:

                joint = cal_extend_data(results.pose_landmarks)
                angle = cal_gesture(joint)

                if sec == 0:
                    timer()
                # print(counter)

                else:
                    angle = np.append(angle, 3)
                    angle = np.expand_dims(angle, axis=0)
                    if first:
                        angle_data = angle
                        first = False
                    else:
                        angle_data = np.append(angle_data, angle, axis=0)
                    # print(grab_data.shape)

                    # print(sec, counter)
                print(sec, counter)
            cv2.imshow('Raw Webcam Feed', image)
            if cv2.waitKey(1) & 0xFF == ord('q') or sec == 2:
                np.savetxt("dataset//Bhujasana.csv", angle_data, delimiter=",", fmt="%.5f")
                exit()
                print('저장끝')
                break
