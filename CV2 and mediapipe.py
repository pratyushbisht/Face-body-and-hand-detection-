# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 15:35:49 2022

@author: praty
"""

import cv2
import mediapipe as mp
 
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic


#for adding s styling
#mp_drawing.DrawingSpec(color=(0,255,0),thickness=(2),circle_radius=(2))

cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret,frame = cap.read()
        #recoloring feed
        image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        #making detections
        results = holistic.process(image)
        
        #recoloring feed back to bgr
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        
        #to draw face landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks,mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(0,255,0),thickness=(2),circle_radius=(2)),
                                  mp_drawing.DrawingSpec(color=(255,0,0),thickness=(2),circle_radius=(2)))
        mp_drawing.draw_landmarks(image, results.face_landmarks,mp_holistic.FACEMESH_CONTOURS)
        
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks,mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2))
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks,mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2))
        
        
        cv2.imshow('Raw Webcam Feed', image)
        
        if cv2.waitKey(10) & 0XFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()