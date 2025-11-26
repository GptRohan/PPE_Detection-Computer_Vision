PPE Detection – Computer Vision
This project is a Personal Protective Equipment (PPE) Detection System built using Python, YOLO, and Streamlit. It automatically checks if people are wearing safety helmets and vests in uploaded videos or live webcam feeds.

Features
Upload and analyze safety/compliance videos

Live webcam PPE detection

Colored bounding boxes highlight detected gear or violations

Automatic screenshot saving and violation logging to CSV

Adjustable detection settings (confidence, frame skip)

Project Structure
app.py – Streamlit frontend for user interaction and real-time results

yolo_video.py – Backend AI, detection and tracking logic

YOLO-Weights – Folder containing pretrained YOLO model (.pt file)

requirements.txt – Python dependencies for easy setup

Usage
Clone this repo and install requirements:

text
pip install -r requirements.txt
Place your YOLO model weights in the YOLO-Weights folder.

Run the app:

text
streamlit run app.py
Upload a video or use webcam live.

Demo
Watch my detailed demo video on LinkedIn
Try live on Streamlit Cloud

Model
This project uses a publicly available pretrained YOLO model. For best results, you can train on your own PPE dataset.

Author
Built by Rohan.
Connect on LinkedIn# PPE_Detection-Computer_Vision
