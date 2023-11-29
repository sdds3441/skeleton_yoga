import mediapipe as mp
import cv2
import timeit

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

width = 1280
height = 720

<<<<<<< HEAD
actions = ['Bhujasana', 'Padamasana', 'Tadasana','Trikasana','Vrikshasana']
cap = cv2.VideoCapture("C://Users//김효찬//Desktop//수업//3학년//2학기//AI프로그래밍//archive//dataset//highlight//Bhujasana//video 1.mp4")
=======
cap = cv2.VideoCapture("C://Users//김효찬//Desktop//수업//3학년//2학기//인공지능프로그래밍//yoga//archive//dataset//test//Bhujasana//video 13.mp4")
>>>>>>> parent of e399180 (first)
cap.set(3, 1280)
cap.set(4, 720)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for yoga_act in actions:
        while cap.isOpened():
            ret, frame = cap.read()

            start_t = timeit.default_timer()

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = holistic.process(image)

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )
            try:
                if results.pose_landmarks.landmark is not None:

                    for i, lm in enumerate(results.pose_landmarks.landmark):

                        if lm.visibility < 0.01:
                            visible = False
                        else:
                            visible = True

                cv2.imshow('Yoga', image)

                terminate_t = timeit.default_timer()

                FPS = int(1. / (terminate_t - start_t))
                #print(FPS)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            except Exception:  # mediapipe 데이터 값이 없을때
                print("화면에 없음",yoga_act)

        if yoga_act == 'Bhujasana':
            cap = cv2.VideoCapture(
                "C://Users//김효찬//Desktop//수업//3학년//2학기//AI프로그래밍//archive//dataset//train//Padamasana//video 1.mp4")
        elif yoga_act == 'Padamasana':
            cap = cv2.VideoCapture(
                "C://Users//김효찬//Desktop//수업//3학년//2학기//AI프로그래밍//archive//dataset//train//Tadasana//video 1.mp4")
        elif yoga_act == 'Tadasana':
            cap = cv2.VideoCapture(
                "C://Users//김효찬//Desktop//수업//3학년//2학기//AI프로그래밍//archive//dataset//train//Trikasana//video 1.mp4")
        elif yoga_act == 'Trikasana':
            cap = cv2.VideoCapture(
                "C://Users//김효찬//Desktop//수업//3학년//2학기//AI프로그래밍//archive//dataset//train//Vrikshasana//video 1.mp4")

cap.release()
cv2.destroyAllWindows()
