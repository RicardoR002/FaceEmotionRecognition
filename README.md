# Face Emotion Detection App

A Streamlit-based web application that detects and analyzes emotions in faces using the FER (Facial Expression Recognition) library. The app supports both image upload and webcam capture.

## Features

- Upload images for emotion analysis
- Use webcam to capture and analyze emotions in real-time
- Display bounding boxes around detected faces
- Show confidence scores for each detected emotion
- Detailed emotion analysis with progress bars
- Adjustable settings through sidebar
- Support for multiple faces in a single image

## Setup

1. Clone this repository
2. Create a virtual environment (recommended):
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On Unix or MacOS:
   source .venv/bin/activate
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the App

To run the application, use the following command:
```bash
streamlit run app.py
```

The app will open in your default web browser. You can then:
1. Choose between uploading an image or using your webcam
2. Adjust settings in the sidebar:
   - Show/hide bounding boxes
   - Set minimum confidence threshold
   - Show/hide detailed emotion analysis
3. View the emotion analysis results

## Requirements

- Python 3.7+
- Webcam (optional, for webcam functionality)
- See requirements.txt for package dependencies

## Notes

- The app uses the MTCNN face detector for better accuracy
- Emotion detection works best with clear, well-lit images
- Multiple faces can be detected and analyzed simultaneously 
