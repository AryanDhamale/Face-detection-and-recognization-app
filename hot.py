import cv2 as cv
import numpy as np


img = cv.imread('media/img.jpg') # matrix hiehth * width * color(RBG)
if img is None :
    exit() 

black= np.zeros(img.shape,dtype=np.uint8)


def reScaleFrame(frame,scale=0.75) : 
    height = int(frame.shape[0]*scale)
    width = int(frame.shape[1]*scale)
    diemention = (width,height)

    return cv.resize(frame,diemention,interpolation=cv.INTER_AREA)


def transform(img,x,y) : 
    transMat= np.array([[1,0,x],[0,1,y]],dtype=np.float32)
    (heigth,width) = img.shape[:2] # heigth * width // 

    return cv.warpAffine(img,transMat,(width,heigth))


def rotated(img,angle) : 
    (heigth,width) = img.shape[:2]
    rotedMat = cv.getRotationMatrix2D((width//2,heigth//2),angle,1.0)
    return cv.warpAffine(img,rotedMat,(width,heigth))



gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

ret,thresh = cv.threshold(gray,125,255,cv.THRESH_BINARY)
cv.imshow('thresh',thresh)

canny = cv.Canny(img,125,175)
cv.imshow('canny',canny)

contours_canny,hierarchy = cv.findContours(canny,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

contours_thresh,hierarchy = cv.findContours(thresh,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

black_copy= black.copy()
cv.drawContours(black,contours_thresh,-1,(0,0,255),2)
cv.drawContours(black_copy,contours_canny,-1,(0,0,255),2)


cv.imshow('contour_thresh',black)
cv.imshow('contour_copy',black_copy)

cv.waitKey(0)

