import cv2
import sqlite3

# Kết nối CSDL SQLite
conn = sqlite3.connect('face_data.db')
cursor = conn.cursor()

# Nạp bộ nhận diện & cascade
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    for (x, y, w, h) in faces:
        id_, conf = recognizer.predict(gray[y:y+h, x:x+w])

        if conf < 80:
            cursor.execute("SELECT name FROM users WHERE id = ?", (id_,))
            result = cursor.fetchone()
            name = result[0] if result else "Unknown"
            label = f"{name} ({round(conf, 2)}%)"
            color = (0, 255, 0)
        else:
            label = "Unknown"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
conn.close()
cv2.destroyAllWindows()
