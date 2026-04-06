import cv2
import os

name = input("Enter person name: ")
path = f"dataset/{name}"
os.makedirs(path, exist_ok=True)

cap = cv2.VideoCapture(0)

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

count = 0

print("Press 's' to save")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(gray, 1.1, 4)

    face_crop = None

    if len(faces) > 0:
        x, y, w, h = faces[0]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        face_crop = frame[y:y+h, x:x+w]

    cv2.imshow("Face Capture", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        if face_crop is not None:
            filename = os.path.join(path, f"{count}.jpg")
            cv2.imwrite(filename, face_crop)
            print(f"Saved image {count}")
            count += 1

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
