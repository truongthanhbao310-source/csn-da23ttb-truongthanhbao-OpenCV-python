import cv2
import numpy as np
from PIL import Image
import os

dataset_path = 'dataset'
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []

    for imagePath in imagePaths:
        img = Image.open(imagePath).convert('L')  # grayscale
        img_np = np.array(img, 'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_np)
        for (x, y, w, h) in faces:
            faceSamples.append(img_np[y:y+h, x:x+w])
            ids.append(id)
    return faceSamples, ids

faces, ids = getImagesAndLabels(dataset_path)
recognizer.train(faces, np.array(ids))
recognizer.write('trainer.yml')

print(f"[INFO] Đã huấn luyện xong {len(np.unique(ids))} người dùng.")
