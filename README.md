PPE Detection – Computer Vision
Personal Protective Equipment (PPE) Detection System built with Python, YOLO, and Streamlit. Detects safety helmet and vest compliance in uploaded videos or live webcam feeds.

Features
Upload and analyze construction/safety videos

Real-time live webcam detection

Colored bounding boxes for hardhats, vests, and violations

Automatic screenshot saving and CSV violation logging

Adjustable detection: confidence and frame skip

Project Structure
app.py — Streamlit frontend for UI and controls

yolo_video.py — Backend AI detection and tracking logic

YOLO-Weights/ — Directory for pretrained YOLO model (.pt file)

requirements.txt — Python dependencies

runtime.txt — Specifies Python version for deployment (e.g. python-3.11.9)

Installation & Usage
Clone this repository:

bash
git clone https://github.com/yourusername/PPE_Detection-Computer_Vision.git
cd PPE_Detection-Computer_Vision
Install requirements:

bash
pip install -r requirements.txt
Add your YOLO .pt model file in the YOLO-Weights folder.

To run locally:

bash
streamlit run app.py
Deployment
This project is compatible with Streamlit Cloud.

Make sure requirements.txt and runtime.txt are included

Deploy app.py and add YOLO weights before launch

Demo
Check out my demo video on LinkedIn
Try the live app on Streamlit Cloud

Model
Uses public YOLO pretrained weights. For best accuracy, train on your own PPE dataset.

Author
Rohan
LinkedIn
