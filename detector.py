from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import imutils

face_detector = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
model = load_model('model/output.h5')

camera = cv2.VideoCapture(0)

while True:
    (grabbed, frame) = camera.read()

    frame = imutils.resize(frame, width = 200)
    frame1 = frame.copy()
    frame_copy = frame.copy()

    rects = face_detector.detectMultiScale(frame1, scaleFactor=1.1, 
		minNeighbors=5, minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE)

    for (fX, fY, fW, fH) in rects:
        roi = frame1[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (32, 32))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        (notSmiling, smiling) = model.predict(roi)[0]
        label = "Smiling" if smiling > notSmiling else "Not Smiling"

        cv2.putText(frame_copy, label, (fX, fY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(frame_copy, (fX, fY), (fX + fW, fY + fH),
            (0, 0, 255), 2)

        cv2.imshow("Face", frame_copy)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()