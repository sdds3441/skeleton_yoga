import timeit
import mediapipe as mp
import cv2
import numpy as np
import socket


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
    action = 9

    v1 = joint[[11, 11, 13, 12, 14, 11, 12, 23, 23, 25, 24, 26], :3]  # Parent joint
    v2 = joint[[12, 13, 15, 14, 16, 23, 24, 24, 25, 27, 26, 28], :3]  # Child joint
    v = v2 - v1
    # Normalize v
    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

    # Get angle using arcos of dot product
    angle = np.arccos(np.einsum('nt,nt->n',
                                v[[0, 1, 0, 3, 0, 0, 5, 7, 8, 7, 10], :],
                                v[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], :]))  # [15,]

    angle = np.degrees(angle)  # Convert radian to degree

    joint_data = np.array([angle], dtype=np.float32)
    ret, results, neighbours, dist = knn.findNearest(joint_data, 3)
    idx = int(results[0][0])


    # Draw gesture result
    if idx in rps_gesture.keys():
        action=idx

    return action


mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverAddressPort = ("127.0.0.1", 5052)

rps_gesture = {0: 'Bhujasana', 1: 'Padamasana', 2: 'Tadasana', 3: 'Trikasana', 4: 'Vrikshasana',5:'Exception_1',6:'Exception_2'}
data = []

width = 1280
height = 720

knn = cv2.ml.KNearest_create()
file = np.genfromtxt('dataset//knn_data.csv', delimiter=',')
angle = file[:, :-1].astype(np.float32)
label = file[:, -1].astype(np.float32)
knn.train(angle, cv2.ml.ROW_SAMPLE, label)
cap = cv2.VideoCapture("C://Users//김효찬//Desktop//수업//3학년//2학기//인공지능프로그래밍//yoga//taekgwon//2023-12-03-160714.webm")
cap.set(3, width)
cap.set(4, height)

# Initiate holistic model

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()

        start_t = timeit.default_timer()
        data = []


        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Make Detections
        results = holistic.process(image)

        # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks

        # Recolor image back to BGR for rendering
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

            joint = cal_extend_data(results.pose_landmarks)
            action = cal_gesture(joint)
        print(action)
        data.append(action)
        sock.sendto(str.encode(str(data)), serverAddressPort)
        cv2.imshow('Raw Webcam Feed', image)

        terminate_t = timeit.default_timer()

        FPS = int(1. / (terminate_t - start_t))
        # print(FPS)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
