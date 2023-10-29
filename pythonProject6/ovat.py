
import cv2
import numpy as np


focal_length = 600
real_object_width = 14.0


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:

        face_center_x = x + w // 2
        face_center_y = y + h // 2


        pixel_distance = np.sqrt((frame.shape[1] // 2 - face_center_x) ** 2 + (frame.shape[0] // 2 - face_center_y) ** 2)


        real_distance = (real_object_width * focal_length) / (w)


        cv2.putText(frame, f"Uzaklik: {real_distance:.2f} cm", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)



        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


        cv2.imshow('Face Detection', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()


