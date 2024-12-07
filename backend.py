import cv2
from datetime import datetime
from keras.models import load_model
from keras.utils import load_img, img_to_array
import numpy as np
import os


facemodel=cv2.CascadeClassifier('./Data/Face_training_data/face.xml')
drowsiness_model=load_model('drowsiness_model.h5')



# =======================================================================================================================================
# vid=cv2.VideoCapture("https://192.168.1.9:8080/video")
vid=cv2.VideoCapture("video3.mp4")


while vid.isOpened():
    # Capturing Video frame by frame
    flag, frame=vid.read()

    if flag:
        # Getting Current date & time when each frame produced
        curr_dt=datetime.now()
        curr_dt=curr_dt.strftime("%d%m%y_%H%M%S")
        # Getting Coordinates of face found in frame and representing them with rectangle box in frame with saving each face image concurrently.
        faces=facemodel.detectMultiScale(frame,scaleFactor=1.05,minNeighbors=5,minSize=(30,30))
        
        for (x,y,l,w) in faces:
            crop_face_frame=frame[x:x+l,y:y+w]
            if len(crop_face_frame)>0:
                cv2.imwrite('temp.jpg',crop_face_frame)
                crop_face=load_img('temp.jpg', target_size=(224,224,3))
                crop_face=img_to_array(crop_face)
                crop_face=np.expand_dims(crop_face, axis=0)
                pred=drowsiness_model(crop_face)[0]
                pred=np.argmax(pred)
                # {0: 'Closed', 1:'Open', 2:'no_yawn',3:'yawn'}
                if pred==0 or pred==3:
                    cv2.rectangle(frame,(x,y),(x+l,y+w),(0,0,255),3)
                    cv2.imwrite(f'./Data/collection/Fatigue/{curr_dt}.jpg',crop_face_frame)
            

        # assigning a normal window with name "Image Window"
        cv2.namedWindow("Camera Window",cv2.WINDOW_NORMAL)
        # Showing image in "Image Window"
        cv2.imshow("Camera Window", frame)
        # Getting user keyboard input.
        key=cv2.waitKey(10)
        # if key input is x then window will close
        if key==ord('x'):
            break
        # closing window while clicking window close button
        if cv2.getWindowProperty("Camera Window",cv2.WND_PROP_VISIBLE)<1:
            break

cv2.destroyAllWindows()



# =======================================================================================================================================
# # Cv2 Code for Viewing Images
# img=cv2.imread('./Data/Active Subjects/image__0.jpg')
# # assigning a normal window with name "Image Window"
# cv2.namedWindow("Image Window", cv2.WINDOW_NORMAL)
# # Showing image in "Image Window"
# cv2.imshow("Image Window", img)
# # Freezing window untill user keyboard input.
# k=cv2.waitKey(0)


# =======================================================================================================================================

