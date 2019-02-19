# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 13:54:34 2019

@author: sriniv11
"""

import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cv2.namedWindow('live feed')
cv2.namedWindow('gray live feed')
cap = cv2.VideoCapture(0)


ret, img = cap.read()

while(ret):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for x,y,w,h in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h), (255,0,0),2)
        cv2.rectangle(gray,(x,y),(x+w,y+h), (255,0,0),2)
    #img1 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    cv2.imshow('live feed',img)
    cv2.imshow('gray live feed',gray)
    #plt.show()
    if(cv2.waitKey(1) == 27):
        break

cv2.destroyWindow('live feed')
cv2.destroyWindow('gray live feed')


cap.release()



#img = img[300:,1000:,0]
print(img)

#plt.imshow(img)
#plt.show(img)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()