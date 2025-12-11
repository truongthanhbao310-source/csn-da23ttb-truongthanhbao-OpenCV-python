import cv2
import os
import sqlite3

# Tạo CSDL SQLite
conn = sqlite3.connect('face_data.db')
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL
    )
''')

# Nhập thông tin người dùng
user_id = input('Nhập ID người dùng (số): ')
name = input('Nhập tên người dùng: ')

cursor.execute('INSERT OR REPLACE INTO users (id, name) VALUES (?, ?)', (user_id, name))
conn.commit()
conn.close()

# Tạo thư mục dataset nếu chưa có
os.makedirs('dataset', exist_ok=True)

# Nạp bộ phát hiện khuôn mặt
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)

cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) 

print("[INFO] Đang thu thập dữ liệu khuôn mặt. Nhấn 'q' để dừng...")

count = 0
while True:
    ret, frame = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        cv2.imwrite(f"dataset/User.{user_id}.{count}.jpg", gray[y:y+h, x:x+w])
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.imshow("Collecting Faces", frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 30:
        break

print(f"[INFO] Đã thu thập {count} ảnh cho ID = {user_id}")
cam.release()
cv2.destroyAllWindows()
