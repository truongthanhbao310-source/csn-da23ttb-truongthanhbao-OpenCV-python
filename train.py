import cv2
import os
import numpy as np

dataset_path = "dataset"

recognizer = cv2.face.LBPHFaceRecognizer_create(
    radius=1,
    neighbors=8,
    grid_x=8,
    grid_y=8
)

faces = []
ids = []

for file in os.listdir(dataset_path):
    if file.endswith(".jpg"):
        path = os.path.join(dataset_path, file)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        id_ = int(file.split(".")[1])
        faces.append(img)
        ids.append(id_)

print("[INFO] Training LBPH...")
recognizer.train(faces, np.array(ids))
recognizer.save("trainer.yml")

print("[INFO] Train xong – trainer.yml đã tạo")
