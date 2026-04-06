import cv2
import os
import numpy as np

dataset_path = "dataset"

faces = []
labels = []
names = {}

label_id = 0

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)

    names[label_id] = person_name

    for image_name in os.listdir(person_folder):
        image_path = os.path.join(person_folder, image_name)

        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        detected_faces = face_detector.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in detected_faces:
            face = gray[y:y+h, x:x+w]
            faces.append(face)
            labels.append(label_id)

    label_id += 1

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))
recognizer.save("face_model.yml")

np.save("labels.npy", names)

print("Training complete")
