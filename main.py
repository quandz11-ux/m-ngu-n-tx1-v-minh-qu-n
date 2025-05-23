import cv2
import dlib
import numpy as np
from tensorflow.keras.models import load_model
from imutils import face_utils
import time

# Load model đã huấn luyện
model = load_model('final_model.h5')

# Hàm dự đoán mở/nhắm
def predict_eye_state(eye_img):
    eye_img = cv2.resize(eye_img, (64, 64))
    eye_img = eye_img.astype("float32") / 255.0
    eye_img = np.expand_dims(eye_img, axis=-1)
    eye_img = np.expand_dims(eye_img, axis=0)
    pred = model.predict(eye_img)
    return "open" if pred[0][0] > 0.5 else "closed"

# Các ngưỡng
CLOSED_EYE_FRAMES = 5
counter = 0

# Khởi tạo
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("68_face_landmarks_predictor.dat")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

cap = cv2.VideoCapture(0)
is_alerting = False
while True:
    c = time.time()
    ret, frame = cap.read()
    if not ret: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)
        left_eye = shape[lStart:lEnd]
        right_eye = shape[rStart:rEnd]

        # Cắt vùng ảnh mắt trái/phải
        def crop_eye_region(gray, eye_points, margin=5):
            x, y, w, h = cv2.boundingRect(np.array(eye_points))
            x = max(x - margin, 0)
            y = max(y - margin, 0)
            w = min(w + 2*margin, gray.shape[1] - x)
            h = min(h + 2*margin, gray.shape[0] - y)
            return gray[y:y+h, x:x+w]

        left_eye_img = crop_eye_region(gray, left_eye)
        right_eye_img = crop_eye_region(gray, right_eye)

        if left_eye_img.size == 0 or right_eye_img.size == 0:
            continue

        # Dự đoán
        left_state = predict_eye_state(left_eye_img)
        right_state = predict_eye_state(right_eye_img)

        if left_state == "closed" and right_state == "closed":
            counter += 1
        else:
            counter = 0
            is_alerting = False
        cv2.imshow("Left Eye", left_eye_img)
        cv2.imshow("Right Eye", right_eye_img)
        # Cảnh báo nếu ngủ gật
        if counter >= CLOSED_EYE_FRAMES:
            cv2.putText(frame, "CANH BAO: NGU GAT!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        cv2.putText(frame, f"L:{left_state} R:{right_state}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Phat hien ngu gat (CNN)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
