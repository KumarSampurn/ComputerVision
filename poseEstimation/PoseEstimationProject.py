import cv2
import time
import poseEstimationModule as PME

cap=cv2.VideoCapture(0)
pTime=0
detector=PME.poseDetector()
while True:
    success,img = cap.read()
    img=cv2.flip(img,1)
        
    img=detector.findPose(img,draw=True)
    lmlist=detector.findPosition(img,draw=True)
    if(len(lmlist)!=0):
    # print(lmlist[0])
        cv2.circle(img,(lmlist[0][1],lmlist[0][2]),15,(255,0,0),cv2.FILLED)
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    
    
    cv2.putText(img,str((int)(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,
                (255,0,255),3)
    
    cv2.imshow("Image",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break