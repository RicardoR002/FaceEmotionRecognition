import streamlit as st
import cv2
from deepface import DeepFace
import numpy as np
import time

def process_image(img):
    try:
        # Show loading state
        with st.spinner('Analyzing emotions...'):
            # Perform emotion analysis
            results = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
            
            # Display results
            for face in results:
                bbox = face['region']
                emotion = face['dominant_emotion']
                emotion_confidence = face['emotion'][emotion]

                if show_bbox:
                    cv2.rectangle(img, (bbox['x'], bbox['y']), 
                                (bbox['x']+bbox['w'], bbox['y']+bbox['h']), 
                                (0, 255, 0), 2)

                label = f"{emotion}"
                if show_confidence:
                    label += f": {emotion_confidence:.2f}"

                cv2.putText(img, label, (bbox['x'], bbox['y']-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            return img
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return img

def main():
    st.title("Face Emotion Recognition App")
    st.markdown("""
    This app uses DeepFace to detect and analyze emotions in faces. You can either upload an image or use your webcam.
    """)

    # Sidebar elements
    st.sidebar.title("Settings")
    detection_confidence = st.sidebar.slider("Detection Confidence", 0.0, 1.0, 0.5, 0.1)
    global show_bbox, show_confidence
    show_bbox = st.sidebar.checkbox("Show Bounding Box", True)
    show_confidence = st.sidebar.checkbox("Show Confidence", True)

    # Input method selection
    input_method = st.radio("Choose input method:", ("Upload Image", "Use Webcam"))

    if input_method == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            try:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if img is None:
                    st.error("Error: Could not read the image file")
                    return
                processed_img = process_image(img)
                st.image(processed_img, channels="BGR")
            except Exception as e:
                st.error(f"Error processing uploaded file: {str(e)}")

    else:  # Use Webcam
        img_file_buffer = st.camera_input("Take a picture")
        if img_file_buffer is not None:
            try:
                # Read image from buffer
                bytes_data = img_file_buffer.getvalue()
                img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    st.error("Error: Could not read the webcam image")
                    return
                processed_img = process_image(img)
                st.image(processed_img, channels="BGR")
            except Exception as e:
                st.error(f"Error processing webcam image: {str(e)}")

if __name__ == "__main__":
    main()