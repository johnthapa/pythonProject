import cv2
from random import randrange

# Load some pre-trained data on face
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# source image video
# img = cv2.imread('aj.jpg')
webcam = cv2.VideoCapture('cute_baby.mp4')

# Iterate forever over frames
while True:
    # Read the current frames
    successful_frame_read, frame = webcam.read()

    # must convert to grayscale
    grayscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscale_img)

    # Draw rectangle around the faces
    for(x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 2)

    cv2.imshow('clever programmer', frame)
    key = cv2.waitKey(1)

    # Q key for quit
    if key == 81 or key == 113:
        break

# release video camera
webcam.release()

print("code complete")