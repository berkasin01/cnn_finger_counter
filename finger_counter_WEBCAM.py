import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("models/howmanyfingersmodel.h5")

# Class labels
classes = ["1 finger", "2 fingers", "3 fingers", "4 fingers",  "5 fingers"]

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    resized = cv2.resize(frame, (256, 256))

    normalised = resized / 255.0

    input_img = np.expand_dims(normalised, 0)

    prediction = model.predict(input_img, verbose=0)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100


    label = f"{classes[predicted_class]} ({confidence:.1f}%)"
    cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imshow("Finger Counter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()