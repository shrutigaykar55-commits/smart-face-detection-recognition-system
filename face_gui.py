import cv2
import numpy as np
import csv
from datetime import datetime
import tkinter as tk
from tkinter import Label, Button, Frame
from PIL import Image, ImageTk

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Face Recognition System")
        self.root.geometry("1000x700")
        self.root.configure(bg="#1e1e1e")

        self.header = Label(
            root,
            text="Smart Face Recognition System",
            font=("Segoe UI", 22, "bold"),
            bg="#1e1e1e",
            fg="white"
        )
        self.header.pack(pady=20)

        self.video_label = Label(root, bg="#2b2b2b")
        self.video_label.pack(pady=20)

        self.button_frame = Frame(root, bg="#1e1e1e")
        self.button_frame.pack(pady=20)

        self.start_btn = Button(
            self.button_frame,
            text="Start",
            font=("Segoe UI", 14, "bold"),
            width=15,
            bg="#4CAF50",
            fg="white",
            command=self.start_camera
        )
        self.start_btn.grid(row=0, column=0, padx=10)

        self.stop_btn = Button(
            self.button_frame,
            text="Stop",
            font=("Segoe UI", 14, "bold"),
            width=15,
            bg="#f39c12",
            fg="white",
            command=self.stop_camera
        )
        self.stop_btn.grid(row=0, column=1, padx=10)

        self.exit_btn = Button(
            self.button_frame,
            text="Exit",
            font=("Segoe UI", 14, "bold"),
            width=15,
            bg="#e74c3c",
            fg="white",
            command=root.quit
        )
        self.exit_btn.grid(row=0, column=2, padx=10)

        self.cap = None
        self.running = False

        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read("face_model.yml")

        self.names = np.load("labels.npy", allow_pickle=True).item()

        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def mark_attendance(self, name):
        with open("attendance.csv", "a", newline="") as file:
            writer = csv.writer(file)
            now = datetime.now()
            writer.writerow([name, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")])

    def start_camera(self):
        if not self.running:
            self.cap = cv2.VideoCapture(0)
            self.running = True
            self.update_frame()

    def stop_camera(self):
        self.running = False
        if self.cap:
            self.cap.release()

    def update_frame(self):
        if self.running:
            ret, frame = self.cap.read()

            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                faces = self.face_detector.detectMultiScale(gray, 1.1, 4)

                for (x, y, w, h) in faces:
                    face = gray[y:y+h, x:x+w]

                    label, confidence = self.recognizer.predict(face)

                    if confidence < 125:
                        name = self.names[label]
                        color = (0, 255, 0)
                        self.mark_attendance(name)
                    else:
                        name = "Unknown"
                        color = (0, 0, 255)

                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

                    cv2.putText(
                        frame,
                        f"{name} | {int(confidence)}",
                        (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2
                    )

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                img = Image.fromarray(frame)
                img = img.resize((850, 500))
                imgtk = ImageTk.PhotoImage(image=img)

                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

            self.video_label.after(10, self.update_frame)

root = tk.Tk()
app = FaceRecognitionApp(root)
root.mainloop()
