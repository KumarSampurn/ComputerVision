import cv2
import mediapipe as mp
import time

class FaceMeshDetector():
    
    def __init__(self, staticMode=False, maxFaces=2, refine_Landmarks=False, minDetectionCon=0.5, minTrackCon=0.5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.refineLdms = refine_Landmarks
        self.minDetectionCon = float(minDetectionCon)
        self.minTrackCon = float(minTrackCon)

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFaces,self.refineLdms ,self.minDetectionCon, self.minTrackCon)
        self.drawSpecs = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    def findFaceMesh(self, img, draw=True):
        self.imgRBG = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRBG) 
        faces=[]
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS, self.drawSpecs, self.drawSpecs)
                
                face=[]
                for id, lms in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lms.x * iw), int(lms.y * ih)
                    # print(id, x, y)
                    face.append([x,y])
                faces.append(face)
            
        return img,faces
    
    
def main():
    cap = cv2.VideoCapture("faceMesh/videos/5.mp4")
    pTime = 0
    detector = FaceMeshDetector()
    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        img,faces = detector.findFaceMesh(img)
         
         
        if (len(faces)):
            print((faces[0]))
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS : {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)    
        cv2.imshow("Image", img)   
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    

if __name__ == "__main__":
    main()