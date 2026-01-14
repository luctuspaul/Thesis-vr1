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

# Initialize Dlib Face Detector
print("[INFO] Loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    './dlib_shape_predictor/shape_predictor_68_face_landmarks.dat')

# ===============================
# INITIALIZE YOLOv4-TINY DETECTOR
# ===============================
if ENABLE_YOLO:
    try:
        config_file = f'yolov{USE_YOLO_VERSION}.cfg'
        weights_file = f'yolov{USE_YOLO_VERSION}.weights'

        print(f"[DEBUG] Loading YOLO config: {config_file}, weights: {weights_file}")
        yolo_detector = YOLODetector(
            config_path=config_file,
            weights_path=weights_file,
            names_path='coco.names',
            confidence_threshold=0.5,
            nms_threshold=0.4
        )
        print(f"[INFO] YOLOv{USE_YOLO_VERSION} detector loaded successfully")
    except Exception as e:
        print(f"[WARNING] Could not load YOLO detector: {e}")
        ENABLE_YOLO = False

# ===============================
# INITIALIZE CAMERA
# ===============================
print("[INFO] Initializing camera...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

frame_width = 1024
frame_height = 576

# ===============================
# FACE LANDMARK AND DROWSINESS SETUP
# ===============================
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = (49, 68)

EYE_AR_THRESH = 0.25
MOUTH_AR_THRESH = 0.79
EYE_AR_CONSEC_FRAMES = 3
COUNTER = 0

image_points = np.array([
    (359, 391),  # Nose tip
    (399, 561),  # Chin
    (337, 297),  # Left eye left corner
    (513, 301),  # Right eye right corner
    (345, 465),  # Left mouth corner
    (453, 469)   # Right mouth corner
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

    # Run YOLO every N frames
    if ENABLE_YOLO and frame_count % YOLO_DETECTION_INTERVAL == 0:
        try:
            # Detect only CELL PHONES
            yolo_detections, _ = yolo_detector.detect_specific_classes(
                frame, target_classes=['cell phone']
            )
        except Exception as e:
            print(f"[ERROR] YOLO detection failed: {e}")
            yolo_detections = []

    # Draw YOLO detections
    if ENABLE_YOLO and yolo_detections:
        for detection in yolo_detections:
            x, y, w, h = detection['box']
            label = detection['label']
            confidence = detection['confidence']
            color = (0, 255, 0) if label == 'cell phone' else (0, 255, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label}: {confidence:.2f}", (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display warning if cellphone detected
        if any(d['label'] == 'cell phone' for d in yolo_detections):
            cv2.putText(frame, "⚠️ WARNING: Phone Detected!", (350, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

    # Detect faces
    rects = detector(gray, 0)
    if len(rects) > 0:
        cv2.putText(frame, f"{len(rects)} face(s) found", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Loop over face detections
    for rect in rects:
        (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH), (0, 255, 0), 1)
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Compute EAR
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0

        cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)

        if ear < EYE_AR_THRESH:
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                cv2.putText(frame, "Eyes Closed!", (500, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            COUNTER = 0

        # Mouth detection
        mouth = shape[mStart:mEnd]
        mar = mouth_aspect_ratio(mouth)
        cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, (0, 255, 0), 1)
        cv2.putText(frame, f"MAR: {mar:.2f}", (650, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if mar > MOUTH_AR_THRESH:
            cv2.putText(frame, "Yawning!", (800, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Draw landmarks + index numbers
        for (i, (x, y)) in enumerate(shape):
            if i == 33:
                image_points[0] = np.array([x, y], dtype='double')
                color = (0, 255, 0)
            elif i == 8:
                image_points[1] = np.array([x, y], dtype='double')
                color = (0, 255, 0)
            elif i == 36:
                image_points[2] = np.array([x, y], dtype='double')
                color = (0, 255, 0)
            elif i == 45:
                image_points[3] = np.array([x, y], dtype='double')
                color = (0, 255, 0)
            elif i == 48:
                image_points[4] = np.array([x, y], dtype='double')
                color = (0, 255, 0)
            elif i == 54:
                image_points[5] = np.array([x, y], dtype='double')
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)

            cv2.circle(frame, (x, y), 1, color, -1)
            cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        # Head Pose estimation
        (head_tilt_degree, start_point, end_point, end_point_alt) = getHeadTiltAndCoords(size, image_points, frame_height)
        cv2.line(frame, start_point, end_point, (255, 0, 0), 2)
        cv2.line(frame, start_point, end_point_alt, (0, 0, 255), 2)
        if head_tilt_degree:
            cv2.putText(frame, f"Head Tilt: {head_tilt_degree[0]:.2f}", (170, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Show window
    cv2.imshow("Driver Drowsiness + Phone Detection", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

    frame_count += 1

# Cleanup
cv2.destroyAllWindows()
vs.stop()
