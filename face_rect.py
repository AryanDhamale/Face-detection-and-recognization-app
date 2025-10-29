import os
import cv2 as cv
import numpy as np

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

characters_name=['kang-han-na','ko-moon-young']

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_rec_model.yml')

def face_recognization(frame) : 
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    
    face_rects = face_cascade.detectMultiScale(gray,1.1,3)

    for (x,y,w,h) in face_rects : 
        face_reco = gray[y:y+h,x:x+w]
        if face_reco is None or face_reco.shape==0 : 
            break

        face_reco = cv.resize(face_reco, (200, 200),interpolation=cv.INTER_AREA)

        label,confidance =  face_recognizer.predict(face_reco)
        
        if confidance < 100 : 
            text = characters_name[label]+str(f" con:{confidance:.2f}")
        else : 
            text=f'Unknown con:{confidance:.2f}'

        cv.putText(frame,text,(x,y-10),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

    return frame
        


def live_detection() : 
    capture = cv.VideoCapture(0)
    
    if not capture.isOpened() : 
        print('failed to load video')
        return
    
    while True : 
        isTrue,frame = capture.read()
        
        if not isTrue : 
            print('loading failed!')
            break

        frame =  face_recognization(frame)
        cv.imshow('video',frame)

        if cv.waitKey(3) & 0xff == ord('w') : 
            break

    capture.release()
    cv.destroyAllWindows()


live_detection()