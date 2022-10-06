import cv2
import numpy as np
import time
import Pose_module as pm

# cap = cv2.VideoCapture("AI trainer\Training videos\curls.mp4")
# cap = cv2.VideoCapture("./Training videos/curls.mp4")
cap = cv2.VideoCapture(0)
detector = pm.poseDetector()
count = 0
dirn = 0
pTime = 0

while True:
    success, img = cap.read()
    # img = cv2.resize(img, (1280, 720))

    # img = cv2.imread("Training videos/curl.jpg")
    img = detector.findPose(img, False)
    lmList = detector.getPositons(img, False)

    if  len(lmList)!=0:
        #left arm
        angle = detector.findAngle(img, 11, 13, 15)
        per = np.interp(angle ,(50, 155), (0,100))
        # bar = np.interp(angle, (220,310), (650,100))
        # print(angle, ",",per)
        #right arm
        # detector.findAngle(img, 12, 14, 16)

        #check for the dumbell curls
        if per == 100:
            if dirn == 0:
                count+=0.5
                dirn = 1
        if per ==0:
            if dirn ==1:
                count+=0.5
                dirn = 0
        
        print(count)

        cv2.rectangle(img, (0,450), (250,720), (32,64,241), cv2.FILLED)
        cv2.putText(img, f'{str(int(count))}', (45,670), cv2.FONT_HERSHEY_PLAIN, 15, (24,123,32), 15)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, f'{str(int(fps))}', (50,100), cv2.FONT_HERSHEY_PLAIN, 5, (24,123,32), 5)

    cv2.imshow("Image", img)
    cv2.waitKey(1)