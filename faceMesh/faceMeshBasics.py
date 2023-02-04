import cv2
import mediapipe as mp
import time

cap= cv2.VideoCapture("faceMesh/videos/6.mp4")
pTime=0


mpDraw= mp.solutions.drawing_utils
mpFaceMesh=mp.solutions.face_mesh
faceMesh=mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpecs=mpDraw.DrawingSpec(thickness=1,circle_radius=1)

while True:
    success , img = cap.read()
    img=cv2.flip(img,1)
    imgRBG=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=faceMesh.process(imgRBG) 
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img,faceLms,mpFaceMesh.FACEMESH_CONTOURS,drawSpecs,drawSpecs)
            
            for id,lms in enumerate(faceLms.landmark):
                ih,iw,ic = img.shape
                x,y=int(lms.x * iw) , int(lms.y * ih)
                print(id,x,y)

    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img,f'FPS : {int(fps)}',(20,70),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),2)    
    cv2.imshow("Image",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
          break
    