import cv2
import mediapipe as mp
import time

cap= cv2.VideoCapture("faceDetection/videos/1.mp4")
pTime=0


mpFaceDetection = mp.solutions.face_detection
faceDetection=mpFaceDetection.FaceDetection(0.75)
mpDraw= mp.solutions.drawing_utils


while True:
    _,img=cap.read()
    cv2.flip(img,1)
    
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=faceDetection.process(imgRGB)
    print(results)
        
    if results.detections:
        for id, detection in enumerate(results.detections):
            # print(detection.location_data.relative_bounding_box) 
            # print(id,detection)
            # print(detection.score)
            # mpDraw.draw_detection(img,detection) 
            bboxc = detection.location_data.relative_bounding_box
            ih,iw,ic=img.shape
            bbox=int(bboxc.xmin * iw),int(bboxc.ymin*ih),int(bboxc.width*iw),int(bboxc.height*ih)
            cv2.rectangle(img,bbox,(255,0,255),2)
            cv2.putText(img,str((int)(detection.score[0]*100))+"%",(bbox[0],bbox[1]-20),cv2.FONT_HERSHEY_PLAIN,1.5,(255,0,255),2)
        
    
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    
    
    cv2.putText(img,str((int)(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(0,255,255),3)
    
    cv2.imshow("Image", img)
    
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
          break