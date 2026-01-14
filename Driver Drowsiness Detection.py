from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import imutils
import time
import dlib
import cv2
import numpy as np
from EAR import eye_aspect_ratio
from MAR import mouth_aspect_ratio
from HeadPose import getHeadTiltAndCoords
from Yolodetector import YOLODetector

# ===============================
# CONFIGURATION
# ===============================
ENABLE_YOLO = True
YOLO_DETECTION_INTERVAL = 30
USE_YOLO_VERSION = "4-tiny"

# ===============================
# THRESHOLDS
# ===============================
EYE_AR_THRESH = 0.25
MOUTH_AR_THRESH = 0.85
EYE_AR_CONSEC_FRAMES = 3

FPS = 30
YAWN_CONSEC_FRAMES = 3 * FPS

COUNTER = 0
YAWN_COUNTER = 0

# ===============================
# LOAD DLIB
# ===============================
print("[INFO] Loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    './dlib_shape_predictor/shape_predictor_68_face_landmarks.dat')

# ===============================
# YOLO
# ===============================
if ENABLE_YOLO:
    try:
        yolo_detector = YOLODetector(
            config_path=f'yolov{USE_YOLO_VERSION}.cfg',
            weights_path=f'yolov{USE_YOLO_VERSION}.weights',
            names_path='coco.names',
            confidence_threshold=0.5,
            nms_threshold=0.4
        )
        print("[INFO] YOLO loaded successfully")
    except Exception as e:
        print("[WARNING] YOLO failed:", e)
        ENABLE_YOLO = False

# ===============================
# CAMERA
# ===============================
print("[INFO] Initializing camera...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

frame_width = 1024
frame_height = 576

# ===============================
# LANDMARKS
# ===============================
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = (49, 68)

image_points = np.array([
    (359, 391),
    (399, 561),
    (337, 297),
    (513, 301),
    (345, 465),
    (453, 469)
], dtype="double")

frame_count = 0
yolo_detections = []

# ===============================
# MAIN LOOP
# ===============================
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=frame_width)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    size = gray.shape

    # ---------------- YOLO ----------------
    if ENABLE_YOLO and frame_count % YOLO_DETECTION_INTERVAL == 0:
        try:
            yolo_detections, _ = yolo_detector.detect_specific_classes(
                frame, target_classes=['cell phone']
            )
        except:
            yolo_detections = []

    if ENABLE_YOLO and yolo_detections:
        for detection in yolo_detections:
            x, y, w, h = detection['box']
            label = detection['label']
            confidence = detection['confidence']
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, f"{label}: {confidence:.2f}",
                        (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        if any(d['label'] == 'cell phone' for d in yolo_detections):
            cv2.putText(frame, " WARNING: Phone Detected!", (350,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 3)

    # ---------------- FACE ----------------
    rects = detector(gray, 0)

    if len(rects) > 0:
        cv2.putText(frame, f"{len(rects)} face(s) found",
                    (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

    for rect in rects:
        (bX,bY,bW,bH) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame,(bX,bY),(bX+bW,bY+bH),(0,255,0),1)

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # ----------- EAR -----------
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        ear = (eye_aspect_ratio(leftEye)+eye_aspect_ratio(rightEye))/2.0

        cv2.drawContours(frame,[cv2.convexHull(leftEye)],-1,(0,255,0),1)
        cv2.drawContours(frame,[cv2.convexHull(rightEye)],-1,(0,255,0),1)

        cv2.putText(frame,f"EAR: {ear:.2f}",(500,50),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2)

        if ear < EYE_AR_THRESH:
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                cv2.putText(frame,"Eyes Closed!",(500,20),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
        else:
            COUNTER = 0

        # ----------- MAR (FIXED LOGIC) -----------
        mouth = shape[mStart:mEnd]
        mar = mouth_aspect_ratio(mouth)

        cv2.drawContours(frame,[cv2.convexHull(mouth)],-1,(0,255,0),1)
        cv2.putText(frame,f"MAR: {mar:.2f}",(650,20),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

        if mar >= MOUTH_AR_THRESH:
            YAWN_COUNTER += 1
        else:
            YAWN_COUNTER = max(0, YAWN_COUNTER - 1)

        if YAWN_COUNTER >= YAWN_CONSEC_FRAMES//2:
            cv2.putText(frame,"Yawning!",(800,20),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

        # ----------- LANDMARKS -----------
        for (i,(x,y)) in enumerate(shape):
            if i in [33,8,36,45,48,54]:
                image_points[[33,8,36,45,48,54].index(i)] = np.array([x,y],dtype="double")
                color=(0,255,0)
            else:
                color=(0,0,255)

            cv2.circle(frame,(x,y),1,color,-1)
            cv2.putText(frame,str(i+1),(x-10,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.35,color,1)

        # ----------- HEAD POSE -----------
        head_tilt,start_point,end_point,end_point_alt = getHeadTiltAndCoords(size,image_points,frame_height)
        cv2.line(frame,start_point,end_point,(255,0,0),2)
        cv2.line(frame,start_point,end_point_alt,(0,0,255),2)

        if head_tilt:
            cv2.putText(frame,f"Head Tilt: {head_tilt[0]:.2f}",(170,20),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)

    cv2.imshow("Driver Drowsiness + Phone Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame_count += 1

# ===============================
# CLEANUP
# ===============================
cv2.destroyAllWindows()
vs.stop()
