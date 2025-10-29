import cv2 as cv
import numpy as np 


# reactable // why gray // minNeighbors // scale

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

def forImage(path) : 
    img = cv.imread(path)
    if img is None : 
        print('failed to load')
        return
    gray=  cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    face_rect =  face_cascade.detectMultiScale(gray,1.1,5)

    for (x,y,w,h) in face_rect : 
        cv.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

    cv.imshow('face-in-image',img)
    cv.waitKey(0)

def forVideo() : 
    capture = cv.VideoCapture(0)
    if not capture.isOpened() : 
        print('failed to capture')
        return
    
    while True : 
        isTrue,frame = capture.read()
        
        if not isTrue : 
            print('failed to capture-frame')
            break
        
        face_rect  = face_cascade.detectMultiScale(frame,1.1,5) 

        for (x,y,w,h) in face_rect : 
            cv.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        
        cv.imshow('video',frame)

        if cv.waitKey(20) & 0xff == ord('w') : 
            break

    capture.release()
    cv.destroyAllWindows()



