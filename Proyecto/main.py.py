#LIBRARIES
import cv2
import numpy as np
#HAARCASCADE FILES, REPLACE YOUR SOURCE
face_cascade = cv2.CascadeClassifier('C:/Users/braya/opencv/sources/data/haarcascades_cuda/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:/Users/braya/opencv/sources/data/haarcascades_cuda/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('C:/Users/braya/opencv/sources/data/haarcascades_cuda/haarcascade_smile.xml')
pen_cascade = cv2.CascadeClassifier('C:/Users/braya/opencv/sources/data/haarcascades_cuda/haarcascade_pen.xml')
banana_cascade = cv2.CascadeClassifier('C:/Users/braya/opencv/sources/data/haarcascades_cuda/haarcascade_banana.xml')
car_cascade = cv2.CascadeClassifier("C:/Users/braya/opencv/sources/data/haarcascades_cuda/haarcascade_cars.xml")

#OPEN WEBCAM (0 IS WEBCAM, YOU CAN REPLACE 0 FOR OTHER NUMBER IF YOU HAVE ANOTHER WEBCAM OR SOME VIDEO)
#EXAMPLE cam = cv2.VideoCapture(1), cam = cv2.VideoCapture(video.mp4)
cam = cv2.VideoCapture(0)

#FONT FOR PUT TEXT
font = cv2.FONT_HERSHEY_SIMPLEX

#WHILE FROM DETECT BANANAS, PENS, CARS, FACES, SMILES, EYES
while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, 1.9, 1)
    for (x,y,w,h) in cars:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.putText(img, 'Carro', (x,y), font, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    pens = pen_cascade.detectMultiScale(gray,  scaleFactor=1.24,minNeighbors=5,minSize=(30, 30),flags = cv2.CASCADE_SCALE_IMAGE)
    for(x,y,w,h) in pens:
        cv2.rectangle(img,(x,y), (x+w, y+h), (255,255,0), 2)
        cv2.putText(img, 'Pluma', (x,y), font, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
    bananas = banana_cascade.detectMultiScale(gray, scaleFactor=1.24,minNeighbors=5,minSize=(30, 30),flags = cv2.CASCADE_SCALE_IMAGE)
    for(x,y,w,h) in bananas:
        cv2.rectangle(img,(x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(img, 'Banana', (x,y), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y), (x+w, y+h), (255,0,0), 2)
        cv2.putText(img, 'Cara', (x,y), font, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for(ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)
            cv2.putText(roi_color, 'Ojo', (ex, ey), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        smiles = smile_cascade.detectMultiScale(roi_gray,scaleFactor = 1.7,minNeighbors = 22,minSize = (25, 25),flags = cv2.CASCADE_SCALE_IMAGE)
        for(zx,zy,zw,zh) in smiles:
            cv2.rectangle(roi_color, (zx,zy), (zx+zw, zy+zh), (0,0,255), 2)
            cv2.putText(roi_color, 'Sonrisa', (zx, zy), font, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    
    #SHOW IMAGE OF WEBCAM IN SCREEN
    cv2.imshow("img", img)
    #IF YOU PRESS ESC BUTTON, EXIT
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
#DESTROY ALL WINDOWS CREATED BY OPENCV
cv2.destroyAllWindows()