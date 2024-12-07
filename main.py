import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.models import load_model
import os
import tempfile
from datetime import datetime

facemodel=cv2.CascadeClassifier('./Data/Face_training_data/face.xml')
try:
    drowsiness_model=load_model('drowsiness_model.h5')
except:
    print("Drowsiness Model not Found...")
    st.rerun()
nav_page=st.sidebar.selectbox("NAVIGATION",options=("Home", "Image", "Video","Camera"))

if nav_page=="Home":
    image_addr='https://kajabi-storefronts-production.kajabi-cdn.com/kajabi-storefronts-production/file-uploads/blogs/22606/images/5481e13-3da0-b8e5-f87f-a5ff1b6da72c_eyeSight_-_Driver_Monitoring_Driver_Asleep_1.jpeg'
    st.title("Drowsiness Detection System")
    st.image(image_addr,width=570)


elif nav_page=="Image":
    st.title("Drowsiness Detection System")
    image=st.file_uploader("Upload Image")
    if image:
        # Changing file into binary format
        b=image.getvalue()
        # Creating buffer for binary format file
        d=np.frombuffer(b,np.uint8)
        # Changing binary file into viewable format
        img=cv2.imdecode(d,cv2.IMREAD_COLOR)

        faces=facemodel.detectMultiScale(img,scaleFactor=1.1,minNeighbors=5,minSize=(30,30))
        curr_dt=datetime.now()
        curr_dt=curr_dt.strftime("%d%m%y_%H%M%S")
        
        for (x,y,l,w) in faces:
            crop_face_frame=img[x:x+l,y:y+w]
            if len(crop_face_frame)>0:
                cv2.imwrite('temp.jpg',crop_face_frame)
                crop_face=load_img('temp.jpg', target_size=(224,224,3))
                crop_face=img_to_array(crop_face)
                crop_face=np.expand_dims(crop_face, axis=0)
                pred=drowsiness_model(crop_face)[0]
                pred=np.argmax(pred)
                print(pred)
                # {0: 'Closed', 1:'Open', 2:'no_yawn',3:'yawn'}
                if pred==0 or pred==3:
                    # Creating red box around detected face to represent Drowsiness.
                    cv2.rectangle(img,(x,y),(x+l,y+w),(0,0,255),3)
                    cv2.imwrite(f'./Data/collection/Fatigue/{curr_dt}.jpg',crop_face_frame)
                else:
                    # Creating green box around detected face if prediction by model other than 0 or 3
                    cv2.rectangle(img,(x,y),(x+l,y+w),(2,240,66),3)
        st.image(img,channels='BGR',width=600)

elif nav_page=="Video":
    st.title("Drowsiness Detection System")
    file=st.file_uploader("Upload Video")
    drowsiness_model=load_model('drowsiness_model.h5')

    # Creating window for showing the frames of video
    window=st.empty()
    if file:
        stop=st.button("Stop")
        tfile=tempfile.NamedTemporaryFile()
        tfile.write(file.read())
        vid=cv2.VideoCapture(tfile.name)
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
                            # Creating red box around detected face to represent Drowsiness.
                            cv2.rectangle(frame,(x,y),(x+l,y+w),(0,0,255),3)
                            cv2.imwrite(f'./Data/collection/Fatigue/{curr_dt}.jpg',crop_face_frame)
                window.image(frame,channels='BGR')
            
elif nav_page=="Camera":
    st.title("Drowsiness Detection System")
    link=st.text_input("Enter IP Camera Link (https://xxx.xx.xxx:xxx) or value 0 for device webcam")
    link=link+'/video'
    if link=='0':
        link=0
    
    # Creating window for showing the frames of video
    window=st.empty()
    vid=cv2.VideoCapture(link)
    stop=st.button("Stop")
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
                
            window.image(frame,channels='BGR')
            if stop:
                window.image(None)
                if link!="":
                    st.rerun()