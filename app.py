import streamlit as st
import cv2
import numpy as np
from fer import FER
from PIL import Image
import io

def process_image(image, detector, show_boxes=True, min_confidence=0.5):
    """Process the image and detect emotions"""
    try:
        # Convert PIL Image to OpenCV format
        if isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Detect emotions
        emotions = detector.detect_emotions(image)
        
        if not emotions:
            st.warning("No faces detected in the image. Please try another image.")
            return image, []
        
        # Draw boxes and labels
        for face in emotions:
            box = face.get('box', None)
            emotions_dict = face.get('emotions', {})
            
            if box and emotions_dict:
                x, y, w, h = box
                
                # Draw bounding box if enabled
                if show_boxes:
                    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Get dominant emotion
                dominant_emotion = max(emotions_dict.items(), key=lambda x: x[1])
                emotion_name, confidence = dominant_emotion
                
                if confidence >= min_confidence:
                    # Add label
                    label = f"{emotion_name}: {confidence:.2f}"
                    cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        return image, emotions
    
    except Exception as e:
        st.error(f"Error in emotion detection: {str(e)}")
        return image, []

def main():
    st.set_page_config(page_title="Face Emotion Detection", layout="wide")
    
    st.title("Face Emotion Detection")
    st.markdown("""
    This app uses the FER (Facial Expression Recognition) library to detect and analyze emotions in faces.
    Upload an image or use your webcam to get started!
    """)
    
    # Sidebar settings
    st.sidebar.title("Settings")
    show_boxes = st.sidebar.checkbox("Show Bounding Boxes", True)
    min_confidence = st.sidebar.slider("Minimum Confidence", 0.0, 1.0, 0.5, 0.1)
    show_detailed = st.sidebar.checkbox("Show Detailed Emotions", True)
    
    # Initialize FER detector
    @st.cache_resource
    def get_detector():
        return FER(mtcnn=True)
    
    detector = get_detector()
    
    # Input method selection
    input_method = st.radio("Choose input method:", ("Upload Image", "Use Webcam"))
    
    if input_method == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            try:
                # Read image
                image = Image.open(uploaded_file)
                
                # Process image
                processed_image, emotions = process_image(image, detector, show_boxes, min_confidence)
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Processed Image")
                    st.image(processed_image, channels="BGR")
                
                if show_detailed and emotions:
                    with col2:
                        st.subheader("Detailed Analysis")
                        for i, face in enumerate(emotions):
                            emotions_dict = face.get('emotions', {})
                            if emotions_dict:
                                st.write(f"Face {i+1}:")
                                for emotion_name, score in emotions_dict.items():
                                    if score >= min_confidence:
                                        st.progress(score, text=f"{emotion_name}: {score:.2f}")
                                st.markdown("---")
                
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
    
    else:  # Webcam
        img_file_buffer = st.camera_input("Take a picture")
        if img_file_buffer is not None:
            try:
                # Read image from buffer
                bytes_data = img_file_buffer.getvalue()
                image = Image.open(io.BytesIO(bytes_data))
                
                # Process image
                processed_image, emotions = process_image(image, detector, show_boxes, min_confidence)
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Processed Image")
                    st.image(processed_image, channels="BGR")
                
                if show_detailed and emotions:
                    with col2:
                        st.subheader("Detailed Analysis")
                        for i, face in enumerate(emotions):
                            emotions_dict = face.get('emotions', {})
                            if emotions_dict:
                                st.write(f"Face {i+1}:")
                                for emotion_name, score in emotions_dict.items():
                                    if score >= min_confidence:
                                        st.progress(score, text=f"{emotion_name}: {score:.2f}")
                                st.markdown("---")
                
            except Exception as e:
                st.error(f"Error processing webcam image: {str(e)}")

if __name__ == "__main__":
    main() 
