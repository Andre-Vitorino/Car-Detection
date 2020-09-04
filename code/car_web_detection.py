# Imports needed
import cv2

# Creating classifier variable
car_classifier = cv2.CascadeClassifier('cascades/cars.xml')

# Reading the first image from webcam
image = cv2.VideoCapture(0)

# This loop do:
#  Reading and detect cars in the video
# To stop it, press 'q' key

while True:
    ok, frame = image.read()

    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    car_detection = car_classifier.detectMultiScale(gray_image, minNeighbors= 5, scaleFactor=1.05, minSize=(20, 20))

    for (x, y, w, h) in car_detection:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow('Car', frame)

    if cv2.waitKey(1) == ord('q'):
        break

# These commands kill all the camera process
cv2.destroyAllWindows()

image.release()
