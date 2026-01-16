import cv2
import numpy as np
import mediapipe as mp
import time

# ===============================
# MEDIAPIPE SETUP
# ===============================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ===============================
# CAMERA
# ===============================
cap = cv2.VideoCapture(0)

# ===============================
# LANDMARK INDEXES
# ===============================
NOSE_TIP = 1

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 291, 13, 14]

# ===============================
# DROWSINESS CONFIG
# ===============================
EAR_THRESHOLD = 0.16
EAR_DANGER_THRESHOLD = 0.07
EAR_TIME_THRESHOLD = 2.0

YAWN_COUNT_THRESHOLD = 3
YAWN_TIME_WINDOW = 120
YAWN_RESET_TIME = 60

# ===============================
# DISTRACTION CONFIG
# ===============================
LOOK_DOWN_COUNT_THRESHOLD = 5
LOOK_DOWN_TIME_WINDOW = 10

# ===============================
# CAMERA BLOCK DETECTION CONFIG
# ===============================
CAMERA_DARK_THRESHOLD = 25
CAMERA_BRIGHT_THRESHOLD = 230
CAMERA_STD_THRESHOLD = 8

# ===============================
# STATE VARIABLES
# ===============================
ear_below_start_time = None
EAR_DROWSY = False

yawn_timestamps = []
YAWN_DROWSY = False
yawn_drowsy_start_time = None

eyes_closed_start_time = None
EYES_CLOSED_DANGER = False

look_down_timestamps = []
DISTRACTED_WARNING_ACTIVE = False

DROWSY_WARNING_ACTIVE = False
CAMERA_BLOCKED = False

# ===============================
# HELPER FUNCTIONS
# ===============================
def euclidean(p1, p2):
    return np.linalg.norm(p1 - p2)

def eye_aspect_ratio(pts, idx):
    v1 = euclidean(pts[idx[1]], pts[idx[5]])
    v2 = euclidean(pts[idx[2]], pts[idx[4]])
    h = euclidean(pts[idx[0]], pts[idx[3]])
    return (v1 + v2) / (2.0 * h + 1e-6)

def mouth_aspect_ratio(pts, idx):
    h = euclidean(pts[idx[0]], pts[idx[1]])
    v = euclidean(pts[idx[2]], pts[idx[3]])
    return v / (h + 1e-6)

# ===============================
# MAIN LOOP
# ===============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    h, w = frame.shape[:2]

    # ===============================
    # CAMERA BLOCK CHECK
    # ===============================
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    std_brightness = np.std(gray)

    CAMERA_BLOCKED = (
        mean_brightness < CAMERA_DARK_THRESHOLD or
        mean_brightness > CAMERA_BRIGHT_THRESHOLD or
        std_brightness < CAMERA_STD_THRESHOLD
    )

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    ear = 0
    mar = 0
    gaze_text = "No Face"

    if result.multi_face_landmarks and not CAMERA_BLOCKED:
        face_landmarks = result.multi_face_landmarks[0]

        pts = np.array([
            (int(lm.x * w), int(lm.y * h))
            for lm in face_landmarks.landmark
        ])

        # ===============================
        # GAZE DIRECTION
        # ===============================
        left_eye_corner = pts[33]
        right_eye_corner = pts[263]
        nose = pts[NOSE_TIP]

        eye_center = (left_eye_corner + right_eye_corner) // 2
        gaze_vec = nose - eye_center
        gaze_vec = gaze_vec / (np.linalg.norm(gaze_vec) + 1e-6)

        if gaze_vec[0] > 0.3:
            gaze_text = "Looking Right"
        elif gaze_vec[0] < -0.3:
            gaze_text = "Looking Left"
        elif gaze_vec[1] > 0.25:
            gaze_text = "Looking Down"
        else:
            gaze_text = "Looking Forward"

        # ===============================
        # DISTRACTION
        # ===============================
        if gaze_text == "Looking Down":
            if not look_down_timestamps or (current_time - look_down_timestamps[-1]) > 0.8:
                look_down_timestamps.append(current_time)

        look_down_timestamps = [
            t for t in look_down_timestamps
            if (current_time - t) <= LOOK_DOWN_TIME_WINDOW
        ]

        DISTRACTED_WARNING_ACTIVE = len(look_down_timestamps) >= LOOK_DOWN_COUNT_THRESHOLD

        # ===============================
        # EAR
        # ===============================
        left_ear = eye_aspect_ratio(pts, LEFT_EYE)
        right_ear = eye_aspect_ratio(pts, RIGHT_EYE)
        ear = min(left_ear, right_ear)

        if ear <= EAR_DANGER_THRESHOLD:
            if eyes_closed_start_time is None:
                eyes_closed_start_time = current_time
            elif (current_time - eyes_closed_start_time) >= 1.0:
                EYES_CLOSED_DANGER = True
        else:
            eyes_closed_start_time = None
            EYES_CLOSED_DANGER = False

        if ear <= EAR_THRESHOLD:
            if ear_below_start_time is None:
                ear_below_start_time = current_time
            elif (current_time - ear_below_start_time) >= EAR_TIME_THRESHOLD:
                EAR_DROWSY = True
        else:
            ear_below_start_time = None
            EAR_DROWSY = False

        # ===============================
        # MAR / YAWNING
        # ===============================
        mar = mouth_aspect_ratio(pts, MOUTH)

        if mar > 0.60:
            if not yawn_timestamps or (current_time - yawn_timestamps[-1]) > 2:
                yawn_timestamps.append(current_time)

        yawn_timestamps = [
            t for t in yawn_timestamps
            if (current_time - t) <= YAWN_TIME_WINDOW
        ]

        if len(yawn_timestamps) >= YAWN_COUNT_THRESHOLD:
            if not YAWN_DROWSY:
                YAWN_DROWSY = True
                yawn_drowsy_start_time = current_time

        if YAWN_DROWSY and yawn_drowsy_start_time:
            if (current_time - yawn_drowsy_start_time) >= YAWN_RESET_TIME:
                YAWN_DROWSY = False
                yawn_drowsy_start_time = None
                yawn_timestamps.clear()

        DROWSY_WARNING_ACTIVE = EAR_DROWSY or YAWN_DROWSY

        for p in pts:
            cv2.circle(frame, tuple(p), 1, (255, 255, 255), -1)

    # ===============================
    # GUI
    # ===============================
    cv2.putText(frame, f"Gaze: {gaze_text}", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.putText(frame, f"EAR: {ear:.2f}", (30, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.putText(frame, f"MAR: {mar:.2f}", (30, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    # ---- PRIORITY ALERTS ----
    if CAMERA_BLOCKED:
        cv2.putText(frame, "!!!! CAMERA BLOCKED",
                    (30, 210),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.1, (255, 0, 255), 4)

    elif EYES_CLOSED_DANGER:
        cv2.putText(frame, "!!!! Danger Driver Eyes Closed",
                    (30, 210),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 255), 4)

    elif DISTRACTED_WARNING_ACTIVE:
        cv2.putText(frame, "!!!! WARNING Driver Distracted",
                    (30, 210),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 165, 255), 3)

    elif DROWSY_WARNING_ACTIVE:
        cv2.putText(frame, "!!!! Warning Drowsy Driving",
                    (30, 210),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 0, 255), 3)

    cv2.imshow("Driver Monitoring System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
