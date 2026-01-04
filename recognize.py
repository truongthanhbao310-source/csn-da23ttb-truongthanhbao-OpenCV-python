import cv2
import sqlite3

# ================== KẾT NỐI DATABASE ==================
conn = sqlite3.connect("face_data.db")
cursor = conn.cursor()

def get_name(user_id):
    cursor.execute("SELECT name FROM users WHERE id=?", (user_id,))
    row = cursor.fetchone()
    return row[0] if row else "Unknown"

# ================== LOAD MODEL & CASCADE ==================
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

face_front = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
face_profile = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_profileface.xml"
)

# ================== CAMERA ==================
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

print("[INFO] Face Recognition started")

THRESHOLD = 70   # ngưỡng nhận diện

# ================== VÒNG LẶP ==================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ================== PHÁT HIỆN MẶT ==================
    faces = []

    # 1. Mặt thẳng
    faces_front = face_front.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100)
    )
    faces.extend(faces_front)

    # 2. Mặt nghiêng phải
    faces_profile = face_profile.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
    )
    faces.extend(faces_profile)

    # 3. Mặt nghiêng trái (flip)
    gray_flip = cv2.flip(gray, 1)
    faces_profile_flip = face_profile.detectMultiScale(
        gray_flip, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
    )

    for (x, y, w, h) in faces_profile_flip:
        x_original = gray.shape[1] - x - w
        faces.append((x_original, y, w, h))

    # ================== NHẬN DIỆN ==================
    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]

        try:
            id_, conf = recognizer.predict(face_img)
        except:
            continue

        # ===== SO SÁNH VỚI TRAIN =====
        if conf < THRESHOLD:
            name = get_name(id_)
            color = (0, 255, 0)   # xanh – có trong train
        else:
            name = "Unknown"
            color = (0, 0, 255)   # đỏ – không có trong train

        label = f"{name} ({round(conf,1)})"

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # ================== HIỂN THỊ ==================
    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ================== GIẢI PHÓNG ==================
cap.release()
cv2.destroyAllWindows()
conn.close()
