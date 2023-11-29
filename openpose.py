# fashion_pose.py : MPII를 사용한 신체부위 검출
import cv2
import timeit

# MPII에서 각 파트 번호, 선으로 연결될 POSE_PAIRS
BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
              "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
              "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
              "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}

POSE_PAIRS = [["Nose", "Neck"], ["Nose", "REye"], ["Nose", "LEye"], ["Neck", "RShoulder"], ["Neck", "LShoulder"],
              ["Neck", "RHip"], ["Neck", "LHip"], ["RShoulder", "RElbow"], ["RElbow", "RWrist"],
              ["LShoulder", "LElbow"],
              ["LElbow", "LWrist"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["LKnee", "LAnkle"], ["LHip", "LKnee"],
              ["REye", "REar"], ["LEye", "LEar"]]
# 각 파일 path
protoFile = "pose_deploy_linevec.prototxt"
weightsFile = "pose_iter_440000.caffemodel"

# 위의 path에 있는 network 모델 불러오기
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

# 쿠다 사용 안하면 밑에 이미지 크기를 줄이는게 나을 것이다
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA) #벡엔드로 쿠다를 사용하여 속도향상을 꾀한다
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA) # 쿠다 디바이스에 계산 요청


###카메라랑 연결...?
capture = cv2.VideoCapture(
    "C://Users//김효찬//Desktop\수업//3학년//2학기//AI프로그래밍//archive//dataset//test//Bhujasana//video 13.mp4")  # 카메라 정보 받아옴
# capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640) #카메라 속성 설정
# capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # width:너비, height: 높이

inputWidth = 1280
inputHeight = 720
inputScale = 1.0 / 255

# 반복문을 통해 카메라에서 프레임을 지속적으로 받아옴
while cv2.waitKey(1) < 0:  # 아무 키나 누르면 끝난다.
    # 웹캠으로부터 영상 가져옴
    hasFrame, frame = capture.read()
    start_t = timeit.default_timer()
    # 영상이 커서 느리면 사이즈를 줄이자
    # frame=cv2.resize(frame,dsize=(320,240),interpolation=cv2.INTER_AREA)

    # 웹캠으로부터 영상을 가져올 수 없으면 웹캠 중지
    if not hasFrame:
        cv2.waitKey()
        break

    #
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    inpBlob = cv2.dnn.blobFromImage(frame, inputScale, (inputWidth, inputHeight), (0, 0, 0), swapRB=False, crop=False)

    imgb = cv2.dnn.imagesFromBlob(inpBlob)
    # cv2.imshow("motion",(imgb[0]*255.0).astype(np.uint8))

    # network에 넣어주기
    net.setInput(inpBlob)

    # 결과 받아오기
    output = net.forward()

    # 키포인트 검출시 이미지에 그려줌
    points = []
    for i in range(0, 18):
        # 해당 신체부위 신뢰도 얻음.
        probMap = output[0, i, :, :]

        # global 최대값 찾기
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        # 원래 이미지에 맞게 점 위치 변경
        x = (frameWidth * point[0]) / output.shape[3]
        y = (frameHeight * point[1]) / output.shape[2]

        # 키포인트 검출한 결과가 0.1보다 크면(검출한곳이 위 BODY_PARTS랑 맞는 부위면) points에 추가, 검출했는데 부위가 없으면 None으로
        if prob > 0.1:
            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), thickness=-1,
                       lineType=cv2.FILLED)  # circle(그릴곳, 원의 중심, 반지름, 색)
            cv2.putText(frame, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
                        lineType=cv2.LINE_AA)
            points.append((int(x), int(y)))
        else:
            points.append(None)

    # 각 POSE_PAIRS별로 선 그어줌 (머리 - 목, 목 - 왼쪽어깨, ...)
    for pair in POSE_PAIRS:
        partA = pair[0]  # Head
        partA = BODY_PARTS[partA]  # 0
        partB = pair[1]  # Neck
        partB = BODY_PARTS[partB]  # 1

        # partA와 partB 사이에 선을 그어줌 (cv2.line)
        if points[partA] and points[partB]:
            cv2.line(frame, points[partA], points[partB], (0, 255, 0), 2)

    cv2.imshow("Output-Keypoints", frame)
    terminate_t = timeit.default_timer()

    FPS = int(1. / (terminate_t - start_t))
    print(FPS)

capture.release()  # 카메라 장치에서 받아온 메모리 해제
cv2.destroyAllWindows()  # 모든 윈도우 창 닫음
