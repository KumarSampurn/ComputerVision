import cv2
import mediapipe as mp
import time

class FaceDetector():
    def __init__(self,minDetectionCon=0.5):
        self.minDetectionCon=minDetectionCon
        self.mpFaceDetection = mp.solutions.face_detection
        self.faceDetection=self.mpFaceDetection.FaceDetection(self.minDetectionCon)
        self.mpDraw= mp.solutions.drawing_utils


    def findFaces(self,img,draw=True):
        
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results=self.faceDetection.process(imgRGB)
        # print(self.results)
        bboxs =[]    
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                
                bboxc = detection.location_data.relative_bounding_box
                ih,iw,ic=img.shape
                bbox=int(bboxc.xmin * iw),int(bboxc.ymin*ih),int(bboxc.width*iw),int(bboxc.height*ih)
                bboxs.append([id,bbox,detection.score])
                
                if draw :
                    img=self.fancyDraw(img,bbox)
                    
                    cv2.putText(img,str((int)(detection.score[0]*100))+"%",(bbox[0],bbox[1]-20),cv2.FONT_HERSHEY_PLAIN,1.5,(255,0,255),2)
                
        return img, bboxs
    
    def fancyDraw(self,img, box,length=30,thickness=5):
        x,y,w,h=box
        x1,y1 = x+w, y+h
        
        cv2.rectangle(img,box,(255,0,255),1)
        
        #top left
        cv2.line(img,(x,y),(x+length,y),(255,0,255),thickness)
        cv2.line(img,(x,y),(x,y+length),(255,0,255),thickness)
        
        #top right
        cv2.line(img,(x+w,y),(x+w-length,y),(255,0,255),thickness)
        cv2.line(img,(x+w,y),(x+w,y+length),(255,0,255),thickness)
        
        #bottom right
        cv2.line(img,(x1,y1),(x1-length,y1),(255,0,255),thickness)
        cv2.line(img,(x1,y1),(x1,y1-length),(255,0,255),thickness)
        
        #bottom left
        cv2.line(img,(x,y+h),(x+length,y+h),(255,0,255),thickness)
        cv2.line(img,(x,y+h),(x,y-length+h),(255,0,255),thickness)
        
        return img
        
      
def main():
    cap= cv2.VideoCapture("faceDetection/videos/6.mp4")
    pTime=0
    detector=FaceDetector()
    
    while True:
        _,img=cap.read()
        cv2.flip(img,1)
        img,bboxs=detector.findFaces(img)
        
        cTime=time.time()
        fps=1/(cTime-pTime)
        pTime=cTime
             
        cv2.putText(img,str((int)(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(0,255,255),2)
        
        cv2.imshow("Image", img)
        
  
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    

if __name__ == "__main__":
    main()

