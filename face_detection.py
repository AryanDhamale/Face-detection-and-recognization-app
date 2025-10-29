import os 
import cv2 as cv
import numpy as np

root_path = os.getcwd()
characters_name=['kang-han-na','ko-moon-young','tejas']
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

features=[]
labels=[]

def get_feature_and_labels() : 
    folder_path = os.path.join(root_path,'train_over_images')
    print('start traning -------')

    for person in os.listdir(folder_path) : 
        #person= kang-han-na 
        for image in os.listdir(os.path.join(folder_path,person)) :
            img = cv.imread(os.path.join(folder_path,person,image))
            if img is None : 
                print('failed during image loading!')
                return
            gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
            
            face_rect = face_cascade.detectMultiScale(gray,1.1,4)

            for (x,y,w,h) in face_rect : 
                face = gray[y:y+h,x:x+w]
                face = cv.resize(face,(200,200),interpolation=cv.INTER_AREA)
                features.append(face)
                labels.append(characters_name.index(person))

    print('end traning -------')        



get_feature_and_labels()

labels_np = np.array(labels)
features_np = np.array(features,dtype=np.object_)

# For OpenCV-contrib (ensure you have opencv-contrib-python installed)
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.train(features_np,labels_np)
face_recognizer.save('face_rec_model.yml')