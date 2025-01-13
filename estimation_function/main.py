import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import visualization as visual
import os

st.title("Pig Weight Estimation App")

# Sidebar options
st.sidebar.title("Upload Options")
app_mode = st.sidebar.selectbox("Choose the input type", ["Upload Image", "Upload Video", "Use Webcam"])

# Session State
if 'current_frame' not in st.session_state:
    st.session_state.current_frame = None
if 'captured_frame' not in st.session_state:
    st.session_state.captured_frame = None
if 'video_playing' not in st.session_state:
    st.session_state.video_playing = False
if 'frame_saved' not in st.session_state:
    st.session_state.frame_saved = False  

# Captured Frames
SAVE_DIR = "captured_frames"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# Process for frame
def process_frame(frame):
    resized_frame = cv2.resize(frame, (640, 640))
    processed_frame = visual.combined_detection(resized_frame)
    return processed_frame

# Function to save the captured frame
def save_frame(frame, filename="captured_frame.jpg"):
    save_path = os.path.join(SAVE_DIR, filename)
    cv2.imwrite(save_path, frame)
    st.session_state.frame_saved = True
    st.sidebar.write(f"Frame saved as {save_path}")

# Upload and process image
if app_mode == "Upload Image":
    uploaded_image = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        
        # Perform weight estimation
        output_image = process_frame(image)
        
        # Display result
        st.image(output_image, channels="BGR", caption="Processed Image")

# Upload and process video
elif app_mode == "Upload Video":
    uploaded_video = st.sidebar.file_uploader("Upload a Video", type=["mp4", "mov", "avi", "MOV"])
    
    if uploaded_video is not None:
        # Save the uploaded video 
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        # Open the video with OpenCV
        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()
        capture_button = st.sidebar.button("Capture Frame")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            processed_frame = process_frame(frame)

            # Store the current frame in session state
            st.session_state.current_frame = processed_frame

            # Display the processed video frame
            stframe.image(processed_frame, channels="BGR")

            # Capture frame logic
            if capture_button:
                st.session_state.captured_frame = st.session_state.current_frame
                if st.session_state.captured_frame is not None:
                    save_frame(st.session_state.captured_frame, "video_frame.jpg")
                    st.image(st.session_state.captured_frame, channels="BGR", caption="Captured Frame")

        cap.release()

# Use webcam for live processing
elif app_mode == "Use Webcam":
    st.write("Click 'Start' to begin live webcam feed.")
    
    start_button = st.button("Start")
    stop_button = st.button("Stop")
    capture_button = st.button("Capture Frame")

    # Webcam processing
    if start_button:
        cap = cv2.VideoCapture(0)  # 0 is the default webcam
        st.session_state.video_playing = True

        stframe = st.empty()

        while st.session_state.video_playing:
            ret, frame = cap.read()
            if not ret:
                st.write("Failed to capture video.")
                break

            processed_frame = process_frame(frame)

            # Store the current frame in session state
            st.session_state.current_frame = processed_frame

            # Display the frame
            stframe.image(processed_frame, channels="BGR")

            # Capture current frame when button is clicked
            if capture_button:
                st.session_state.captured_frame = st.session_state.current_frame
                if st.session_state.captured_frame is not None:
                    save_frame(st.session_state.captured_frame, "webcam_frame.jpg")
                    st.image(st.session_state.captured_frame, channels="BGR", caption="Captured Frame")

            # Stop live feed when 'Stop' button is pressed
            if stop_button:
                st.session_state.video_playing = False
                break

        cap.release()

st.sidebar.write("Select an option to start estimating weight.")
