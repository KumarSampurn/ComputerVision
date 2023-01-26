import cv2
import mediapipe as mp
import time

cap= cv2.VideoCapture("faceDetection/videos/1.mp4")
pTime=0
while True:
    _,img=cap.read()
    
    cv2.flip(img,1)
    
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    
    
    cv2.putText(img,str((int)(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
    
    cv2.imshow("Image", img)
    
    
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
          break