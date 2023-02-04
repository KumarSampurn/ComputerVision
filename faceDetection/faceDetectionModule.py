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
                
                
                cv2.rectangle(img,bbox,(255,0,255),2)
                cv2.putText(img,str((int)(detection.score[0]*100))+"%",(bbox[0],bbox[1]-20),cv2.FONT_HERSHEY_PLAIN,1.5,(255,0,255),2)
            
        return img, bboxs
    
      
      
def main():
    cap= cv2.VideoCapture("faceDetection/videos/2.mp4")
    pTime=0
    detector=FaceDetector()
    
    while True:
        _,img=cap.read()
        cv2.flip(img,1)
        img,bboxs=detector.findFaces(img)
        
        
        cTime=time.time()
        fps=1/(cTime-pTime)
        pTime=cTime
        
        
        cv2.putText(img,str((int)(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(0,255,255),3)
        
        cv2.imshow("Image", img)
        
        
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    


if __name__ == "__main__":
    main()

