import numpy as np
import random
import cv2
import tensorflow as tf
import streamlit as st
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode
from PIL import Image
from keras.models import Sequential
from keras.layers import InputLayer,Conv2D,MaxPooling2D,Dropout,Flatten,Dense
import io

classifier=Sequential()

classifier.add(InputLayer(input_shape=(48,48,1),dtype="float32",sparse=False,ragged=False,name="conv2d_input"))
classifier.add(Conv2D(name="conv2d",filters=32,kernel_size=(3,3),activation="relu"))
classifier.add(Conv2D(name="conv2d_1",filters=64,kernel_size=(3,3),activation="relu"))
classifier.add(MaxPooling2D(name="max_pooling2d",pool_size=(2,2)))
classifier.add(Dropout(name="dropout",rate=0.25))
classifier.add(Conv2D(name="conv2d_2",filters=128,kernel_size=(3,3),activation="relu"))
classifier.add(MaxPooling2D(name="max_pooling2d_1",pool_size=(2,2)))
classifier.add(Conv2D(name="conv2d_3",filters=128,kernel_size=(3,3),activation="relu"))
classifier.add(MaxPooling2D(name="max_pooling2d_2",pool_size=(2,2)))
classifier.add(Dropout(name="dropout_1",rate=0.25))
classifier.add(Flatten(name="flatten"))

classifier.add(Dense(name="dense",units=1024,activation="relu"))
classifier.add(Dense(name="dense_1",units=720,activation="relu"))
classifier.add(Dropout(name="dropout_2",rate=0.5))

classifier.add(Dense(name="dense_2",units=480,activation="relu"))
classifier.add(Dropout(name="dropout_3",rate=0.5))
classifier.add(Dense(name="dense_3",units=240,activation="relu"))
classifier.add(Dense(name="dense_4",units=5,activation="softmax"))

model_json = classifier.to_json()
with open("emotion_model.json", "w") as json_file:
    json_file.write(model_json)


# load model
emotion_dict = {0:'angry', 1 :'happy', 2: 'neutral', 3:'sad', 4: 'surprise'}
# load json and create model
json_file = open('emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
classifier = model_from_json(loaded_model_json)

# load weights into new model
classifier.load_weights("MFER/emotion_model1.h5")

#load face
try:
    face_cascade = cv2.CascadeClassifier('MFER/haarcascade_frontalface_default.xml')
except Exception:
    st.write("Error loading cascade classifiers")

class Faceemotion:
    def transform(self, picture):
        img = picture

        #image gray
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            image=img_gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img=img, pt1=(x, y), pt2=(
                x + w, y + h), color=(255, 0, 0), thickness=2)
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                finalout = emotion_dict[maxindex]
                output = str(finalout)
            label_position = (x, y)
            cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img,output

def main():
    # Face Analysis Application #
    st.title("Real Time Music Recommendation system using Face Emotion Detection Application")
    activiteis = ["Home", "Webcam Face Detection", "About"]
    choice = st.sidebar.selectbox("Select Activity", activiteis)
    st.sidebar.markdown(
        """ Developed by Trishal Reddy    
            """)
    if choice == "Home":
        html_temp_home1 = """<div style="background-color:#6D7B8D;padding:10px">
                                            <h4 style="color:white;text-align:center;">
                                            It recommends Music based on facial expressions using OpenCV, Custom CNN model and Streamlit.</h4>
                                            </div>
                                            </br>"""
        st.markdown(html_temp_home1, unsafe_allow_html=True)
        st.write("""
                 The application has three functionalities.

                 1. Real time face detection using web cam.

                 2. Real time face emotion recognization.

                 3. Real time Music recommendation 

                 """)
    elif choice == "Webcam Face Detection":
        st.header("Webcam")
        st.write("Take a picture and detect your face emotion")
        picture = st.camera_input("Give an Expression")
        
        if picture is not None:
            pic=np.array(Image.open(picture))
            model = Faceemotion()
            img,mood=model.transform(pic)
            st.image(pic)
            st.write(f"# Your probably in {mood} mood. So, let me recommend you some music")
            x=random.randint(1,7)
            data=open("MFER/songs/{}/{}_{}.mp3".format(mood,mood,x),'rb')
            st.audio(data,format='mp3')

    elif choice == "About":
        st.subheader("About this app")
        html_temp_about1= """<div style="background-color:#6D7B8D;padding:10px">
                                    <h4 style="color:white;text-align:center;">
                                    Real time face emotion detection application using OpenCV, Custom Trained CNN model and Streamlit.</h4>
                                    </div>
                                    </br>"""
        st.markdown(html_temp_about1, unsafe_allow_html=True)

        html_temp4 = """
                             		<div style="background-color:#98AFC7;padding:10px">
                             		<h4 style="color:white;text-align:center;">This Application is developed by Trishal Reddy using Streamlit Framework, Opencv, Tensorflow and Keras library for demonstration purpose. If you have any suggestion or wnat to comment just write a mail at trishalreddybokka@gmail.com. </h4>
                             		<h4 style="color:white;text-align:center;">Thanks for Visiting</h4>
                             		</div>
                             		<br></br>
                             		<br></br>"""

        st.markdown(html_temp4, unsafe_allow_html=True)

    else:
        pass


if __name__ == "__main__":
    main()
