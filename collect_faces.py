import cv2
import os
import sqlite3

# ===== DB =====
conn = sqlite3.connect('face_data.db')
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL
)
""")

user_id = int(input("Nhập ID: "))
name = input("Nhập tên: ")

cursor.execute("INSERT OR REPLACE INTO users VALUES (?,?)", (user_id, name))
conn.commit()
conn.close()

# ===== DATASET =====
os.makedirs("dataset", exist_ok=True)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

profile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_profileface.xml"
)

cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)

if not cam.isOpened():
    print("❌ Không mở được webcam")
    exit()

count = 0
padding = 30

print("[INFO] Thu thập ảnh – nhấn Q để thoát")

while True:
    ret, frame = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ===== 1. MẶT THẲNG =====
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

    # ===== 2. MẶT NGHIÊNG PHẢI =====
    faces_profile = profile_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
    faces = list(faces) + list(faces_profile)

    # ===== 3. MẶT NGHIÊNG TRÁI (FLIP) =====
    gray_flip = cv2.flip(gray, 1)
    faces_profile_flip = profile_cascade.detectMultiScale(
        gray_flip, 1.1, 5, minSize=(80, 80)
    )

    for (x, y, w, h) in faces_profile_flip:
        # Chuyển tọa độ về ảnh gốc
        x_original = gray.shape[1] - x - w
        faces.append((x_original, y, w, h))

    # ===== LƯU ẢNH =====
    for (x, y, w, h) in faces[:1]:  # mỗi frame lấy 1 mặt
        count += 1

        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)

        face = gray[y1:y2, x1:x2]
        face = cv2.resize(face, (200, 200))

        cv2.imwrite(f"dataset/User.{user_id}.{count}.jpg", face)

        # Lật ảnh bổ sung
        face_flip = cv2.flip(face, 1)
        cv2.imwrite(f"dataset/User.{user_id}.{count}_flip.jpg", face_flip)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Collect Faces", frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 300:
        break

print("[INFO] Đã thu thập", count, "ảnh")
cam.release()
cv2.destroyAllWindows()
