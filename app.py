import streamlit as st
from deepface import DeepFace
import tensorflow as tf
import numpy as np

import tempfile


interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()



@st.cache
def facenet(img1, img2):
    input_data = np.array(img1, dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()
    x = interpreter.get_tensor(output_details[0]['index'])
    
    input_data = np.array(img2, dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()
    y = interpreter.get_tensor(output_details[0]['index'])
    distance = findCosineDistance(x[0], y[0])
    if(distance<=0.4):
        return True, distance
    else:
        return False, distance
@st.cache
def arcface(ipth1, ipth2):
    result = DeepFace.verify(img1_path = ipth1, 
      img2_path = ipth2, 
      model_name = "ArcFace",
      detector_backend = 'mtcnn'
    )
    
    return result['verified'], result['distance']
@st.cache
def findCosineDistance(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

    


st.set_page_config(layout="wide")

images = {
    "Angelina Jolie 1": "test_images/aj1.jpg",
    "Angelina Jolie 2": "test_images/aj2.jpg",
    "Christiano Ronaldo 1": "test_images/cr1.jpg",
    "Christiano Ronaldo 2": "test_images/cr2.jpg",
    "test1": "test_images/1.jpg",
    "test2": "test_images/2.jpg",
    "Take your own Selfie": ""
}

st.markdown("<h1 style='text-align: center; font-size: 100px'>Redefining Mukham's AI</h1>", unsafe_allow_html=True)
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
   st.header("Registered Image")
   option = st.selectbox("Select the Anchor Image", images.keys())
   ipth1 = images[option]
   if option=="Take your own Selfie":
    placeholder = st.empty()
    pic = placeholder.camera_input("", key=1)
    if not pic:
        st.stop()
    else:
        placeholder.empty()
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(pic.read())
        ipth1=tfile.name
    
   img1 = DeepFace.detectFace(img_path = ipth1, target_size = (160, 160), detector_backend = 'mtcnn')
   st.image(img1, use_column_width="always")

with col2:
   st.header("Test Image")
   option2 = st.selectbox("Select the Test Image", images.keys())
   ipth2 = images[option2]
   if option2=="Take your own Selfie":
    placeholder = st.empty()
    pic = placeholder.camera_input("", key=2)
    if not pic:
        st.stop()
    else:
        placeholder.empty()
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(pic.read())
        ipth2=tfile.name
   img2 = DeepFace.detectFace(img_path = ipth2, target_size = (160, 160), detector_backend = 'mtcnn')
   st.image(img2, use_column_width="always")

st.markdown("---")


if st.button('Verify'):

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<h2 style='text-align: center; color: white;'>\"My\" FaceNet</h2>", unsafe_allow_html=True)
        res1 = list(facenet(img1[np.newaxis,:], img2[np.newaxis]))

    with c2:
        st.markdown("<h2 style='text-align: center; color: white;'>\"Your\" ArcFace</h2>", unsafe_allow_html=True)
        res2 = list(arcface(ipth1, ipth2))

    c11, c12, c21, c22 = st.columns(4)

    c11.metric("Verfied", res1[0])
    c12.metric("Distance", round(res1[1],2), "Threshold = 0.4", delta_color="off")

    c21.metric("Verfied", res2[0])
    c22.metric("Distance", round(res2[1],2), "Threshold = 0.68", delta_color="off")

else:
    st.error('C\'mon start the magic!')