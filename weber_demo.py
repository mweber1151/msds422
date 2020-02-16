#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 17:14:05 2019

@author: msweber
"""

import cv2
import face_recognition
import numpy as np
import os
    
known_faces = []
known_encodings = []
for f in os.listdir('C:/Users/msweber/Pictures/Faces/'):
    img = face_recognition.load_image_file('C:/Users/msweber/Pictures/Faces/' + f)
    enc1 = face_recognition.face_encodings(img)[0]
    known_faces.append(f.replace('.jpg','').replace('.png','').replace('.jpeg',''))
    known_encodings.append(enc1)

known_faces


face_cascade = cv2.CascadeClassifier('C:/OpenCV/opencv-master/data/haarcascades/haarcascade_frontalface_alt.xml')

vid = cv2.VideoCapture(0)

while True:
    _,frame = vid.read()
    frame = cv2.resize(frame, (800,800))
   
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for(x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        img = frame[y:y+h, x:x+w]
        img1 = img[:,:,::-1]
        face_locations = face_recognition.face_locations(img1)
        en = face_recognition.face_encodings(img1, face_locations )
        if len(en)> 0:
            matches = face_recognition.compare_faces(known_encodings, en[0])
            if True in matches:
                print(known_faces[matches.index(True)])
            #else:
                #known_faces.append('aaron')
                #known_encodings.append(en[0])
               # cv2.imwrite('/home/allison/Datasets/facial/aaron.jpg', img)
    cv2.imshow("Camera", frame)
    key = cv2.waitKey(1)
    if key ==ord('q') & 0xFF:
        break

cv2.destroyAllWindows()
    

vid.release()